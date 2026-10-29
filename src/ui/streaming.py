"""
Chainlit rendering of one agent turn: the astream_events of the graph become live steps.
Used by: app.py (on_message, one TurnRenderer per question)

One parent step per question, open while the agent works, collapsed once the answer starts.
While the agent works, the parent step is named after the running stage, then after its summary.
Each stage of the graph gets its own step, in the order it runs: named after the action while it
runs ("Écriture de la requête SQL…"), then after its result and duration:

    ▸ SQL Agent · 2 tables lues · 7,3 s
        Question reformulée : « Chiffre d'affaires par mois en 2022 ? » · 0,6 s     only when rewritten
        Contexte de l'agent : 2 tables · métrique revenue · 1,1 s                   vector search + rerank
        Valeur trouvée : « Sao Paulo »                                              only when values were found
        Schéma complet conservé, 14 colonnes
        Plan de requête établi · 1,1 s                                              streamed
        Requête SQL écrite · lit invoice, invoiceline · 2,1 s
        12 lignes retournées · 2,4 s

A rejected query shows as it runs: "Requête rejetée par Hive", "Requête SQL corrigée", then the rows.
"""

import asyncio
import re
import time
import uuid
from functools import cache

import chainlit as cl
import sqlparse
from chainlit.utils import utc_now
from langchain_core.messages import HumanMessage
from loguru import logger

from config import cfg
from lib.column_pruning import get_column_names
from lib.messages import clarify_question, get_question, not_available_reason
from lib.query_lineage import build_query_lineage
from lib.semantics import load_profile, load_semantics
from lib.sql_tools import contains_expression, get_query_tables
from resources import metrics_store, schema_store
from ui.follow_ups import suggest_follow_ups
from ui.result_card import build_result_card_props

FOLLOW_UPS_TIMEOUT_S = 10  # past this, the answer simply comes without suggestions

# Graph node → name of its step while it runs ({engine}: Hive...). Consecutive nodes of a stage
# share its step, nodes absent here have none (format_answer writes the answer)
NODE_STAGES = {
    "condense_question": "Reformulation de la question…",
    "retrieve_schema":   "Recherche des tables utiles…",
    "rerank_tables":     "Recherche des tables utiles…",
    "match_values":      "Recherche des valeurs…",
    "prune_columns":     "Réduction du schéma…",
    "planner":           "Planification de la requête…",
    "sql_coder":         "Écriture de la requête SQL…",
    "sql_validator":     "Exécution sur {engine}…",
}

# Stages shown only once their node ends with something to show (the question was rewritten,
# values were found)
LATE_STAGES = {"condense_question", "match_values"}

# Values named in the title of the match_values step, the others are counted
MAX_TITLE_VALUES = 3

# A question that gets no query (state unavailable_reason): label of the steps, and what they say.
# Business words only, never the name of the withheld column nor any value
UNAVAILABLE_LABELS = {
    "restricted": "information confidentielle",
    "absent":     "information absente des données disponibles",
}
UNAVAILABLE_DETAILS = {
    "restricted": "Cette information n'est pas accessible depuis l'assistant, pour des raisons de "
                  "protection des données : aucune requête n'est exécutée.",
    "absent":     "Aucune colonne des tables disponibles ne contient cette information : "
                  "aucune requête n'est exécutée.",
}


class TurnRenderer:
    """
    Turns the graph events of one question into Chainlit steps and the answer message.
    Holds the whole state of the turn: steps, query result, rejected attempts, answer.
    """

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine  = cfg.get_db(db_name).dialect.capitalize() if db_name else "la base"
        self.run_id  = None   # root run of the graph, for the LangSmith feedback

        self.turn_start  = time.perf_counter()
        self.node_starts = {}  # run_id of a running graph node → start time

        self.agent_step  = None   # parent step
        self.stage       = None   # name of the running stage step
        self.stage_step  = None
        self.stage_start = 0.0

        self.rejected = 0      # queries rejected by the database
        self.result   = None   # query result of the result card

        self.answer_msg            = None
        self.follow_ups_task       = None
        self.clarification_options = []   # options of a clarifying question, shown as the follow-up buttons

    # ── Graph events ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        # Child of Chainlit's "on_message" step, like the answer message: shown before the answer
        current = cl.context.current_step
        self.agent_step = cl.Step(name="SQL Agent", type="run", default_open=True, auto_collapse=True,
                                  parent_id=current.id if current else None)
        self.agent_step.start = utc_now()
        await self.agent_step.send()

    async def handle(self, event: dict) -> None:
        """Route one astream_events (v2) event."""
        kind = event["event"]
        if self.run_id is None:
            self.run_id = event.get("run_id")

        # A graph node run is the chain named after its node (nested runs carry the
        # node name in their metadata only)
        node    = event["metadata"].get("langgraph_node", "")
        is_node = bool(node) and event.get("name") == node

        if kind == "on_chain_start" and is_node:
            self.node_starts[event["run_id"]] = time.perf_counter()
            await self._on_node_start(node)

        elif kind == "on_chain_end" and is_node:
            start    = self.node_starts.pop(event["run_id"], None)
            duration = time.perf_counter() - start if start is not None else None
            output   = event["data"].get("output")
            handler  = getattr(self, f"_end_{node}", None)
            if handler and isinstance(output, dict):
                stage = await handler(event["data"].get("input") or {}, output)
                if stage and node in LATE_STAGES:
                    await self._open_stage(NODE_STAGES[node].format(engine=self.engine))
                if stage and self.stage_step:
                    # A late stage opens as its node ends: its step lasts as long as the node
                    await self._close_stage(*stage, duration=duration if node in LATE_STAGES else None)

        elif kind == "on_chat_model_stream":
            await self._on_token(node, event["data"]["chunk"])

    async def finish(self, status: str = "done") -> None:
        """
        Close the step still open and collapse the parent step under its summary.
        status: "done", "failed" (graph error) or "stopped" (user clicked stop). Idempotent.
        """
        if not self.agent_step or self.agent_step.end:
            return
        if self.stage_step:
            outcome = {"failed": "échec", "stopped": "arrêté"}.get(status)
            title   = self.stage.removesuffix("…")
            await self._close_stage(f"{title} · {outcome}" if outcome else title, error=status == "failed")

        parts = ["SQL Agent"]
        if self.result and self.result["tables"]:
            n = len(self.result["tables"])
            parts.append(f"{n} table{'s' if n > 1 else ''} lue{'s' if n > 1 else ''}")
        if status == "stopped":
            parts.append("arrêté")
        parts.append(_seconds(time.perf_counter() - self.turn_start))

        self.agent_step.name     = " · ".join(parts)
        self.agent_step.is_error = status == "failed"
        self.agent_step.end      = utc_now()
        await self.agent_step.update()

    async def finalize_answer(self) -> None:
        """Attach the result card and the feedback actions to the answer, then the follow-ups."""
        if not self.answer_msg:
            return
        self._attach_result_card()
        if self.run_id:
            # Each button carries the id of the other one, so a vote removes both: one vote per answer
            run_id, up_id, down_id = str(self.run_id), str(uuid.uuid4()), str(uuid.uuid4())
            self.answer_msg.actions = [
                cl.Action(id=up_id,   name="thumbs_up",   payload={"run_id": run_id, "other_id": down_id},
                          icon="thumbs-up",   tooltip="Bonne réponse"),
                cl.Action(id=down_id, name="thumbs_down", payload={"run_id": run_id, "other_id": up_id},
                          icon="thumbs-down", tooltip="Réponse à revoir"),
            ]
        await self.answer_msg.update()
        await self._send_follow_ups()

    # ── Node starts and tokens ────────────────────────────────────────────────

    async def _on_node_start(self, node: str) -> None:
        # The answer starts: the agent's work is over, collapse its step
        if node in ("format_answer", "report_unreachable", "ask_clarification"):
            await self.finish()
        elif node in NODE_STAGES and node not in LATE_STAGES:
            title = NODE_STAGES[node].format(engine=self.engine)
            if self.stage_step and self.stage == title:
                return  # next node of the same stage, its step stays open
            await self._open_stage(title)

    async def _on_token(self, node: str, chunk) -> None:
        content = normalize_chunk_content(chunk.content)
        if not content or chunk.tool_call_chunks:
            return

        # The plan is written live in its step
        if node == "planner" and self.stage_step:
            await self.stage_step.stream_token(content)

        # The answer is written live in its message
        elif node == "format_answer":
            if not self.answer_msg:
                self.answer_msg = cl.Message(content="")
                await self.answer_msg.send()
            await self.answer_msg.stream_token(content)

    # ── Node ends: (final title, detail, error) of the stage step ────────────

    async def _end_condense_question(self, input: dict, output: dict) -> tuple | None:
        """Show the rewritten question, only when the question was actually rewritten."""
        original  = next((m.content for m in reversed(input.get("messages", []))
                          if isinstance(m, HumanMessage)), "")
        condensed = output.get("standalone_question", "")
        if not original or not condensed or original.strip().lower() == condensed.strip().lower():
            return None
        return f"Question reformulée : « {condensed} »", f"**Question :** {original}\n\n**Reformulée :** {condensed}"

    async def _end_rerank_tables(self, input: dict, output: dict) -> tuple:
        candidates = list(input.get("candidate_schemas", {}))
        selected   = list(output.get("selected_schemas", {}))
        chosen     = output.get("llm_selected_tables") or []
        metrics    = output.get("selected_metrics") or []

        if not output.get("is_in_scope", True):
            if output.get("unavailable_reason") == "restricted":
                return "Hors périmètre : information confidentielle", UNAVAILABLE_DETAILS["restricted"]
            return "Aucune table utile : question hors périmètre", f"Aucune table de {self.db_name} ne permet de répondre à cette question."

        # The schema given to planner and sql_coder, not the tables the query reads: the title
        # counts them, only the tables actually read are named (SQL step, parent step, result card)
        title = f"Contexte de l'agent : {len(selected)} table{'s' if len(selected) > 1 else ''}"
        if metrics:
            title += f" · métrique{'s' if len(metrics) > 1 else ''} {', '.join(metrics)}"

        # A state, not a story: the tables given to the agent, each with the reason it is there,
        # then the candidates of the vector search left out
        picked = [t for t in selected if t in chosen]
        added  = [t for t in selected if t not in picked]
        metric_tables = {t for m in metrics for t in metrics_store[self.db_name][m].tables}
        by_metric     = [t for t in added if t in metric_tables]
        by_link       = [t for t in added if t not in metric_tables]   # bridges between two selected tables
        dropped       = [t for t in candidates if t not in selected]

        metric_label = f"la métrique {metrics[0]}" if len(metrics) == 1 else f"les métriques {', '.join(metrics)}"
        reasons = [
            (f"Choisie{'s' if len(picked) > 1 else ''} par l'agent", picked),
            (f"Ajoutée{'s' if len(by_metric) > 1 else ''} pour {metric_label}", by_metric),
            (f"Ajoutée{'s' if len(by_link) > 1 else ''} par les liens entre tables", by_link),
        ]
        if not picked:  # no valid table from the LLM: rerank_tables falls back to all the candidates
            reasons = [("Toutes les candidates", selected)]
        given = "\n".join(f"- {reason} : {', '.join(tables)}" for reason, tables in reasons if tables)
        detail = [f"**Fournies à l'agent ({len(selected)}) :**\n{given}"]
        if dropped:
            detail.append(f"**Écartées ({len(dropped)}) :** {', '.join(dropped)}")
        return title, "\n\n".join(detail)

    async def _end_match_values(self, input: dict, output: dict) -> tuple | None:
        """Show the values of the database found for the question, only when some were found."""
        matched = output.get("matched_values") or {}
        values  = list(dict.fromkeys(v for columns in matched.values() for vs in columns.values() for v in vs))
        if not values:
            return None

        plural = "s" if len(values) > 1 else ""
        title  = f"Valeur{plural} trouvée{plural} : " + ", ".join(f"« {v} »" for v in values[:MAX_TITLE_VALUES])
        if len(values) > MAX_TITLE_VALUES:
            title += f" +{len(values) - MAX_TITLE_VALUES}"

        # Where each value was found: the columns planner can filter on
        detail = "\n".join(
            f"- {table}.{column} : {', '.join(f'`{v}`' for v in vs)}"
            for table, columns in matched.items() for column, vs in columns.items()
        )
        return title, detail

    async def _end_prune_columns(self, input: dict, output: dict) -> tuple:
        before = sum(len(get_column_names(ddl)) for ddl in input.get("selected_schemas", {}).values())
        after  = len(get_column_names(output.get("retrieved_schema", "")))
        if after < before:
            return f"Schéma réduit : {before} → {after} colonnes", ""
        return f"Schéma complet conservé, {before} colonnes", ""

    async def _end_planner(self, input: dict, output: dict) -> tuple:
        # Tables the plan uses beyond the selection, loaded by planner
        # A plan saying "Not available (...)": its reason, never the plan's own words (column names)
        plan = output.get("query_plan", "")
        if reason := not_available_reason(plan.strip().split("\n", 1)[0]):
            return f"Plan : {UNAVAILABLE_LABELS[reason]}", UNAVAILABLE_DETAILS[reason]
        if clarified := clarify_question(plan):
            return ("Deux lectures possibles : question posée",
                    f"**Question :** {clarified[0]}\n\n**Options :** {', '.join(clarified[1]) or '—'}")

        candidates = input.get("selected_schemas", {})
        loaded     = [t for t in output.get("selected_schemas", {}) if t not in candidates]
        detail     = plan
        if loaded:
            detail += f"\n\n**Tables ajoutées par le plan :** {', '.join(loaded)}"
        title = "Plan de requête établi"
        if loaded:
            title += f" · {', '.join(loaded)} ajoutée{'s' if len(loaded) > 1 else ''}"
        return title, detail

    async def _end_sql_coder(self, input: dict, output: dict) -> tuple:
        query  = output.get("generated_query", "")
        if not query:
            reason = output.get("unavailable_reason") or "absent"
            return f"Aucune requête : {UNAVAILABLE_LABELS[reason]}", UNAVAILABLE_DETAILS[reason]
        title  = "Requête SQL corrigée" if input.get("validation_feedback") else "Requête SQL écrite"
        detail = _sql_block(query)
        tables = _tables_in_sql(query, input["selected_db"])
        if tables:
            title += f" · lit {', '.join(tables)}"
            # The tables of the agent's context this query doesn't read
            unused = [t for t in input.get("selected_schemas", {}) if t not in tables]
            if unused:
                detail = f"**Non utilisées du contexte :** {', '.join(unused)}\n\n{detail}"
        return title, detail

    async def _end_sql_validator(self, input: dict, output: dict) -> tuple | None:
        if output.get("validation_feedback"):
            # Only the database error is shown, never the instructions written for sql_coder
            self.rejected += 1
            if output.get("validation_check") == "scope":
                # Caught before execution: tables out of the scope of the agent, never run
                return ("Tables hors du périmètre de l'agent", f"```\n{output.get('validation_error', '')}\n```\n"
                        "La requête n'a pas été exécutée : l'agent la réécrit sans elles.", True)
            if output.get("validation_check") == "excluded":
                # Caught before execution: columns hidden from the agent, never run
                return ("Colonnes non disponibles pour l'agent", f"```\n{output.get('validation_error', '')}\n```\n"
                        "La requête n'a pas été exécutée : l'agent la réécrit sans elles.", True)
            if output.get("validation_check") == "values":
                # Caught before execution: a filtered value the data doesn't hold
                return ("Valeur absente des données", f"```\n{output.get('validation_error', '')}\n```\n"
                        "L'agent corrige le filtre avant d'exécuter la requête.", True)
            error = _short_db_error(output.get("validation_error", ""))
            return f"Requête rejetée par {self.engine}", f"```\n{error}\n```\nL'agent corrige la requête à partir de cette erreur.", True

        if output.get("unavailable_reason") == "restricted":
            # A second query reading excluded columns: the question gets the confidential answer
            return f"Requête non exécutée : {UNAVAILABLE_LABELS['restricted']}", UNAVAILABLE_DETAILS["restricted"]

        if output.get("unavailable_reason") == "absent":
            # A second query reading tables out of the scope: the question gets the out-of-scope answer
            return f"Requête non exécutée : {UNAVAILABLE_LABELS['absent']}", UNAVAILABLE_DETAILS["absent"]

        if output.get("validation_check") == "unreachable":
            # The database could not be reached: the query is not at fault, it was not run
            error = _short_db_error(output.get("validation_error", ""))
            return f"{self.engine} injoignable", f"```\n{error}\n```\nLa requête n'a pas pu être exécutée.", True

        if "result_rows" not in output:
            return None
        rows     = output["result_rows"]
        query    = input.get("generated_query", "")
        question = get_question(input)
        self.result = {
            "columns":   output["result_columns"],
            "rows":      rows,
            "truncated": output.get("result_truncated", False),
            "sql":       query,
            "tables":    _tables_in_sql(query, input["selected_db"]),
            "lineage":   _lineage_of(query, input["selected_db"]),
            "question":  question,
        }
        # Suggest follow-up questions while format_answer writes the answer
        self.follow_ups_task = asyncio.create_task(suggest_follow_ups(
            question = question,
            schema   = input.get("retrieved_schema", ""),
            sql      = query,
            columns  = output["result_columns"],
            rows     = rows,
        ))
        detail = f"**Colonnes :** {', '.join(output['result_columns'])}"
        if self.result["truncated"]:
            detail += f"\n\nSeules les {len(rows)} premières lignes sont lues (limite de sécurité)."
        rows_label = _rows_label(self.result)
        return f"{rows_label} retournée{'s' if rows_label.endswith('s') else ''}", detail

    async def _end_report_unreachable(self, input: dict, output: dict) -> None:
        """Send the fixed answer of an unreachable database (no LLM, nothing streamed)."""
        await self._end_format_answer(input, output)

    async def _end_ask_clarification(self, input: dict, output: dict) -> None:
        """Send the clarifying question (no LLM, nothing streamed); its options become the clickable buttons."""
        self.clarification_options = output.get("clarification_options") or []
        await self._end_format_answer(input, output)

    async def _end_format_answer(self, input: dict, output: dict) -> None:
        """Send the answer when it was not streamed (semantic cache hit)."""
        messages = output.get("messages") or []
        if self.answer_msg or not messages:
            return
        content = normalize_chunk_content(messages[-1].content)
        if content:
            self.answer_msg = cl.Message(content=content)
            await self.answer_msg.send()

    # ── Steps ─────────────────────────────────────────────────────────────────

    async def _open_stage(self, title: str) -> None:
        """Open a stage step, named after its action while it runs."""
        if self.stage_step:  # the previous node ended without a result to show
            await self._close_stage()
        self.stage       = title
        self.stage_start = time.perf_counter()
        self.stage_step  = cl.Step(name=title, type="run", parent_id=self.agent_step.id)
        self.stage_step.start = utc_now()
        await self.stage_step.send()
        # The parent shows the running action while it is collapsed
        self.agent_step.name = title
        await self.agent_step.update()

    async def _close_stage(self, title: str = "", detail: str = "", error: bool = False,
                           duration: float | None = None) -> None:
        """End the running stage step, named after its result and duration."""
        if duration is None:
            duration = time.perf_counter() - self.stage_start
        title = title or self.stage.removesuffix("…")
        step  = self.stage_step
        step.name     = " · ".join(part for part in (title, _took(duration)) if part)
        step.is_error = error
        if detail:
            step.output = detail
        step.end = utc_now()
        await step.update()
        self.stage = self.stage_step = None

    # ── Answer elements ───────────────────────────────────────────────────────

    def _attach_result_card(self) -> None:
        """Attach the ResultCard element (chart / table / SQL) of this turn's query to the answer."""
        if not self.result or not self.result["rows"]:
            return
        props = build_result_card_props(
            columns   = self.result["columns"],
            rows      = self.result["rows"],
            sql       = self.result["sql"],
            truncated = self.result["truncated"],
            tables    = self.result["tables"],
            lineage   = self.result["lineage"],
            retries   = self.rejected,
            db_name   = self.db_name or "",
            question  = self.result["question"],
        )
        self.answer_msg.elements = [cl.CustomElement(name="ResultCard", props=props, display="inline")]

    async def _send_follow_ups(self) -> None:
        """
        Send the FollowUps element (clickable next questions) under the answer already sent.
        Sent on its own once the suggestions are ready, so it never delays the answer and card.
        A clarifying question sends its options instead: one click answers it.
        """
        if self.clarification_options:
            element = cl.CustomElement(name="FollowUps", props={"questions": self.clarification_options}, display="inline")
            self.answer_msg.elements.append(element)
            await element.send(for_id=self.answer_msg.id)
            return
        if not self.follow_ups_task:
            return
        try:
            questions = await asyncio.wait_for(self.follow_ups_task, timeout=FOLLOW_UPS_TIMEOUT_S)
        except asyncio.TimeoutError:
            return
        if not questions:
            return

        element = cl.CustomElement(name="FollowUps", props={"questions": questions}, display="inline")
        self.answer_msg.elements.append(element)
        await element.send(for_id=self.answer_msg.id)


def normalize_chunk_content(content) -> str:
    """
    Normalize chunk content to handle different LLM provider formats.
    Gemini often returns a list of blocks/parts, while OpenAI returns a plain string.
    """
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


# ── Private helpers ───────────────────────────────────────────────────────────


def _short_db_error(error: str, max_len: int = 240) -> str:
    """
    Keep the readable part of a database error. A Hive error comes wrapped by pyhive and
    SQLAlchemy: "(pyhive.exc.OperationalError) TExecuteStatementResp(... errorMessage="Error
    while compiling statement: FAILED: SemanticException [Error 10004]: Line 1:7 Invalid
    table alias or column reference 'x'") [SQL: ...] (Background on this error at: ...)".
    An Impala one by impyla: "(impala.error.HiveServer2Error) AnalysisException: Could not
    resolve column/field reference: 'x'\n[SQL: ...]".
    """
    match = re.search(r"""errorMessage=(['"])(.*?)\1[,)]""", error, re.DOTALL)
    message = match.group(2) if match else error.split("\n[SQL:")[0]
    message = message.replace("\\n", "\n")  # errorMessage comes from a repr: newlines are escaped
    message = re.sub(r"^\([\w.]+\) ", "", message)  # driver exception class: (impala.error.HiveServer2Error)
    # A wrapped error ends with its cause: "... dependencies.\nUnderlying error: org.apache...
    # SemanticException: Line 4:6 Expression not in GROUP BY key 'x'". The cause is the useful part.
    if "Underlying error:" in message:
        message = message.split("Underlying error:")[-1].strip()
        message = re.sub(r"^[\w.]+\.\w+Exception: ", "", message)  # org.apache...SemanticException:
    message = re.sub(r"^Error while compiling statement: FAILED: ", "", message)
    message = re.sub(r"^\w+Exception( \[Error \d+\])?:? ", "", message)   # SemanticException [Error 10004]:
    message = re.sub(r"^[Ll]ine \d+:\d+ ", "", message).strip()            # Line 1:7
    message = message[:1].upper() + message[1:]
    return message if len(message) <= max_len else message[:max_len].rstrip() + "…"


def _sql_block(sql: str) -> str:
    formatted = sqlparse.format(sql, reindent=True, keyword_case="upper")
    return f"```sql\n{formatted}\n```"


def _rows_label(result: dict) -> str:
    """ "5 lignes", "100+ lignes", "aucune ligne" """
    n = len(result["rows"])
    if n == 0:
        return "aucune ligne"
    more = "+" if result["truncated"] else ""
    return f"{n}{more} ligne{'s' if n > 1 or more else ''}"


def _seconds(duration: float) -> str:
    return f"{duration:.1f} s".replace(".", ",")


def _took(duration: float | None) -> str:
    """Duration of a step name ("1,2 s"), empty when unknown or negligible."""
    return _seconds(duration) if duration is not None and duration >= 0.1 else ""


def _tables_in_sql(sql: str, db_name: str) -> list[str]:
    """
    Tables of the database the query actually reads, parsed by sqlglot.
    Matched against every table of the database, not only the selected ones: the SQL may read
    a table planner loaded, or one named by a "-- Joins:" hint.
    """
    known = {table.lower(): table for table in schema_store[db_name]}
    names = get_query_tables(sql, cfg.get_db(db_name).dialect)
    return [known[name.lower()] for name in names if name.lower() in known]


def _lineage_of(sql: str, db_name: str) -> dict | None:
    """
    Lineage of the query (lib/query_lineage.py) for the "Calcul" tab of the result card, its
    unqualified columns resolved with the indexed schemas, each aggregated output named after the
    business metric it computes (metric: its name, or None), with what the semantic layer knows of
    it (context, see _lineage_context). None when it can't be read: the card goes without the tab,
    never without the result.
    """
    dialect = cfg.get_db(db_name).dialect
    try:
        lineage = build_query_lineage(sql, dialect, _schema_columns(db_name))
        if lineage is None:
            return None
        for step in lineage["steps"]:
            # The table each alias of the step stands for (None: another step), for the metric's own tables
            tables = {s["alias"]: s["name"] if s["kind"] == "table" else None for s in step["sources"]}
            for output in step["outputs"]:
                output["metric"] = next((name for name, metric in metrics_store[db_name].items()
                                         if output["aggregated"]
                                         and contains_expression(output["expression"], metric.expr, dialect, tables)), None)
        lineage["context"] = _lineage_context(lineage, db_name)
        return lineage
    except Exception as e:   # a query shape the lineage doesn't handle
        logger.warning(f"[lineage] {type(e).__name__}: {e}")
        return None


@cache
def _schema_columns(db_name: str) -> dict[str, list[str]]:
    """{table: [column, ...]} of the indexed schemas, read once like them."""
    return {table: get_column_names(ddl) for table, ddl in schema_store[db_name].items()}


@cache
def _table_facts(db_name: str) -> dict[str, dict]:
    """
    {table (lowercase): {rows, description, columns: {column (lowercase): description}}} from the
    semantic layer (profile.yaml, tables.yaml), read once like the indexed documents.
    """
    semantics, profile = load_semantics(db_name), load_profile(db_name)
    facts = {}
    for table in {*semantics, *profile}:
        described = semantics.get(table) if isinstance(semantics.get(table), dict) else {}
        measured  = profile.get(table) if isinstance(profile.get(table), dict) else {}
        facts[table.lower()] = {
            "rows":        measured.get("rows"),
            "description": described.get("description"),
            "columns":     {str(c).lower(): d for c, d in (described.get("columns") or {}).items()},
        }
    return facts


def _lineage_context(lineage: dict, db_name: str) -> dict:
    """
    What the semantic layer knows of what the lineage shows, for the graph and its tooltips:
      tables   {table: {rows, description, columns: {column: description}}}, the descriptions of
               the columns the query reads only (never those of a column it doesn't show)
      metrics  {metric: description} of the metrics its outputs compute
    """
    facts, tables = _table_facts(db_name), {}
    for step in lineage["steps"]:
        for source in step["sources"]:
            fact = facts.get(source["name"].lower()) if source["kind"] == "table" else None
            if not fact:
                continue
            entry = tables.setdefault(source["name"], {"rows": fact["rows"], "description": fact["description"], "columns": {}})
            entry["columns"].update({c: d for c in source["columns"] if (d := fact["columns"].get(c.lower()))})
    metrics = {o["metric"]: metrics_store[db_name][o["metric"]].description
               for step in lineage["steps"] for o in step["outputs"] if o.get("metric")}
    return {"tables": tables, "metrics": metrics}
