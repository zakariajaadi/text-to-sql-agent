# offline/eval/run.py
# ===========
# EvaluationRunner — core orchestration for Text-to-SQL benchmarking.
#
# Usage (CLI, from the project root, PYTHONPATH=src — see `make eval`):
#   python -m offline.eval.run --db Northwind                   # datasets/northwind.yaml → results/northwind.json
#   python -m offline.eval.run --difficulty hard                # filter by difficulty
#   python -m offline.eval.run --limit 5                        # run first N only
#   python -m offline.eval.run --db Northwind --rescore         # compare the saved queries again, no LLM
#   python -m offline.eval.run --only PayPal --only BDC         # run these questions again, merged into the saved run
#
# Usage (importable):
#   from offline.eval.run import EvaluationRunner
#   from offline.eval.eval_lib import results_to_df, print_summary
#   runner = EvaluationRunner(db="Chinook")
#   runner.run(difficulty="hard")
#   df = results_to_df(runner.results)
#   print_summary(runner.summary)

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from agent.graph import build_graph
from config import cfg
from offline.eval.eval_lib import (
    DatabaseUnreachable,
    compute_summary,
    compute_table_metrics,
    execution_accuracy,
    extract_tables,
    load_questions,
    load_schema,
    print_summary,
    results_to_df,
    save_results,
    uniform_score,
)
from offline.eval.eval_lib.judge_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
from offline.eval.eval_lib.judge_schemas import JudgeEvaluation
from offline.eval.history import PARTIAL, RESCORE, RUN, record
from resources import db_registry

# Inputs and outputs of the eval, one file per database (Chinook → datasets/chinook.yaml, results/chinook.json)
EVAL_DIR = Path(__file__).resolve().parent


def dataset_path(db: str) -> Path:
    return EVAL_DIR / "datasets" / f"{db.lower()}.yaml"


def results_path(db: str) -> Path:
    """Tracked by git: after a run, `git diff` shows the change against the last committed results."""
    return EVAL_DIR / "results" / f"{db.lower()}.json"


def judged_only(q: dict) -> bool:
    """
    A dataset question with `compare: judge`: several presentations are equally right (labels of
    age bands, an "unknown" group or not), so its rows are not compared, only the judge scores it.
    """
    return q.get("compare") == "judge"


# Technical errors are run again once at the end of the run, after this pause (a cluster or VPN hiccup)
RETRY_ERRORS_AFTER_S = 30


def error_result(q: dict, error: str, detail: str) -> dict:
    """
    Result of a question the run could not score (error: "unreachable" or "exception").
    It counts in no score: compute_summary leaves it out and reports how many there were.
    """
    return {
        "question": q["question"],
        "difficulty": q["difficulty"],
        "category": q["category"],
        "generated_sql": None,
        "selected_tables": [],
        "latency_s": 0,
        "table_recall": 0, "table_precision": 0, "table_hallucination": 0,
        "exec_exact": None,
        "judge": uniform_score(0, f"Technical error: {detail}"),
        "error": error,
    }


# ── EvaluationRunner ──────────────────────────────────────────────────────────

class EvaluationRunner:
    """
    Runs benchmark questions against the Text-to-SQL agent
    and evaluates results with an LLM-as-judge.

    Flow:
      1. __init__         → build graph, load schema, init judge, connect DB
      2. run()            → evaluate each question
      3. results_to_df()  → DataFrame for analysis     (from lib)
      4. print_summary()  → aggregate metrics           (from lib)
      5. save_results()   → persist as JSON             (from lib)
    """

    def __init__(self, db: str = "Chinook"):
        # 1. Build the agent graph (same graph used in production)
        self.db = db
        self.graph = build_graph()

        # 2. Load full DDL schema — given to the judge as context
        self.schema = load_schema(db)

        # 3. Init a separate LLM for judging (avoids self-evaluation bias)
        self.judge = self._init_judge()

        # 4. Connect to database for execution accuracy
        self.db_conn = db_registry[self.db]

        # 5. Placeholders — populated by run()
        self.results: list[dict] = []
        self.summary: dict = {}

    def _init_judge(self):
        """Separate model from agent to avoid self-evaluation bias."""
        self.judge_model = os.getenv("JUDGE_MODEL", "google_genai:gemini-2.5-pro")
        return init_chat_model(self.judge_model, temperature=0).with_structured_output(JudgeEvaluation)

    def meta(self) -> dict:
        """What produced this run, saved with the results: date, models, commit."""
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, cwd=EVAL_DIR, check=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                   text=True, cwd=EVAL_DIR, check=True).stdout.strip()
            commit += "-dirty" if dirty else ""
        except (OSError, subprocess.CalledProcessError):
            commit = None
        return {
            "db": self.db,
            "run_at": datetime.now().isoformat(timespec="minutes"),
            "commit": commit,
            "agent_model": cfg.llm.model,
            "agent_fast_model": cfg.llm_fast.model,
            "judge_model": self.judge_model,
        }


    # ── Core ──────────────────────────────────────────────────────────────────

    def run(self, benchmark: str | Path | None = None,
            difficulty: str = None, limit: int = None, only: list[str] | None = None) -> list[dict]:
        """
        Main entry point — runs the full evaluation pipeline.
        1. Load and filter questions from YAML (only: questions whose text holds one of these words)
        2. For each question: run agent → compute metrics → judge SQL → execution accuracy
        3. Compute aggregate summary
        """
        # 1. Load questions
        questions = load_questions(benchmark or dataset_path(self.db), difficulty)
        if only:
            words = [w.lower() for w in only]
            questions = [q for q in questions if any(w in q["question"].lower() for w in words)]
            if not questions:
                raise SystemExit(f"No question holds {' or '.join(only)}")
        if limit:
            questions = questions[:limit]

        logger.info(f"Running {len(questions)} questions on '{self.db}'")
        self.results = []

        # 2. Evaluate each question
        for i, q in enumerate(questions):
            if i > 0:
                time.sleep(2)  # throttle to avoid rate limits

            logger.info(f"[{i+1}/{len(questions)}] {q['question']}")

            self.results.append(self._evaluate_safely(q))

        # 3. Technical errors (database unreachable, agent crash) say nothing of the answer:
        #    each one is run again once at the end, then left out of the scores if it fails again
        failed = [i for i, r in enumerate(self.results) if r.get("error")]
        if failed:
            logger.warning(f"{len(failed)} technical error(s), running them again in {RETRY_ERRORS_AFTER_S}s")
            time.sleep(RETRY_ERRORS_AFTER_S)
            for i in failed:
                logger.info(f"[retry] {questions[i]['question']}")
                self.results[i] = self._evaluate_safely(questions[i])

        # 4. Aggregate
        self.summary = compute_summary(self.results)
        return self.results

    def _evaluate_safely(self, q: dict) -> dict:
        """_evaluate_one, retried once without SQL; a technical error gives a result marked "error"."""
        try:
            result = self._evaluate_one(q)
            # Retry once if agent failed to generate SQL
            if result["generated_sql"] is None and q["category"] != "out_of_scope":
                logger.warning(f"Retrying: {q['question']}")
                time.sleep(5)
                result = self._evaluate_one(q)
            return result
        except DatabaseUnreachable as e:
            logger.error(f"Database unreachable: {q['question']} — {e}")
            return error_result(q, "unreachable", str(e))
        except Exception as e:
            logger.error(f"Agent failed: {q['question']} — {e}")
            return error_result(q, "exception", f"{type(e).__name__}: {e}")

    def _evaluate_one(self, q: dict) -> dict:
        """
        Evaluate a single benchmark question.
        1. Run agent → get SQL + selected tables
        2. Compute table recall/precision (programmatic)
        3. Execution accuracy — compare result sets against gold queries
        4. LLM judge → 6-dimension scores
        """
        # 1. Run agent (a database it could not reach is a technical error, not its answer)
        agent_res = self._run_agent(
            q["question"],
            tags=["eval", f"difficulty:{q['difficulty']}"]
        )
        if agent_res["unreachable"]:
            raise DatabaseUnreachable("the agent could not run its query")

        # 2. Table metrics
        table_metrics = compute_table_metrics(
            q["expected_tables"], agent_res["selected_tables"]
        )

        # 3. Execution accuracy (not for a question several presentations answer: compare: judge)
        exec_score = None if judged_only(q) else execution_accuracy(
            self.db_conn,
            q.get("accepted_queries", []),
            agent_res["generated_sql"]
        )

        # 4. Judge
        is_oos = q["category"] == "out_of_scope"
        is_refusal = not agent_res["generated_sql"]
        judge_res = self._judge_query(
            q["question"], q.get("accepted_queries", []),
            agent_res["generated_sql"], is_oos, is_refusal,
        )

        return {
            "question": q["question"],
            "difficulty": q["difficulty"],
            "category": q["category"],
            "generated_sql": agent_res["generated_sql"],
            "selected_tables": agent_res["selected_tables"],
            "latency_s": agent_res["latency_s"],
            **table_metrics,
            "exec_exact": exec_score["exact"] if exec_score else None,
            "judge": judge_res,
            **({"compare": "judge"} if judged_only(q) else {}),
        }

    def _run_agent(self, question: str, tags: list[str] = None) -> dict:
        """
        Invoke agent graph and extract results from state.
        1. Invoke graph with db_name config
        2. Read generated SQL from state
        3. Parse selected tables from DDL
        """
        start = time.perf_counter()
        
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)],
             "selected_db": self.db,  # bypass route_database
             },
             config={
                        "configurable": {
                            "db_name": self.db,
                            "thread_id": f"eval_{id(question)}",
                        },
                        "tags": tags or [],
                    }
        )

        return {
            "generated_sql": result.get("generated_query") or None,
            "selected_tables": result.get("llm_selected_tables", []),
            "latency_s": round(time.perf_counter() - start, 2),
            "unreachable": result.get("validation_check") == "unreachable",
        }

    def _judge_query(self, question: str, ground_truth: list[str],
                     generated_sql: str | None,
                     is_out_of_scope: bool, is_refusal: bool) -> dict:
        """
        LLM-as-judge evaluation.
        1. Short-circuit deterministic cases (out-of-scope, missing SQL)
        2. Format ground truth queries
        3. Invoke judge with structured output → JudgeEvaluation
        """
        # 1. Deterministic scoring — no LLM call needed
        if is_out_of_scope and is_refusal:
            return uniform_score(5, "Correctly refused out-of-scope question.")
        if is_out_of_scope and not is_refusal:
            return uniform_score(1, "Generated SQL for out-of-scope question.")
        if not generated_sql:
            return uniform_score(1, "No SQL generated for in-scope question.")

        # 2. Format ground truth
        gt_text = "\n\n".join(
            f"### Accepted Query {i+1}:\n{q}"
            for i, q in enumerate(ground_truth)
        )

        # 3. Invoke judge — returns JudgeEvaluation directly
        response = self.judge.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=JUDGE_USER_PROMPT.format(
                question=question,
                schema=self.schema,
                ground_truth=gt_text,
                generated_sql=generated_sql,
                language=cfg.get_db(self.db).user_language,
            )),
        ])

        return response.model_dump()


# ── Partial rerun ─────────────────────────────────────────────────────────────

def merge_results(saved: list[dict], new: list[dict]) -> list[dict]:
    """The saved results with each new one in place of the saved one of the same question (appended if new)."""
    by_question = {r["question"]: r for r in new}
    merged = [by_question.pop(r["question"], r) for r in saved]
    return merged + list(by_question.values())


# ── Rescore ───────────────────────────────────────────────────────────────────

def rescore(db: str, benchmark: str | Path | None = None, path: str | Path | None = None) -> dict:
    """
    Score the SQL of a saved run again against the database, without calling any LLM:
    after a change of the comparison (execution.py), the agent's queries and the judge's notes stay.
    1. Read the saved results and the accepted queries of the dataset
    2. exec_exact again for each query (a database unreachable keeps the previous score)
    3. Summary again, saved in place
    """
    # 1. Inputs
    path = Path(path or results_path(db))
    data = json.loads(path.read_text())
    gold = {q["question"]: q for q in load_questions(benchmark or dataset_path(db))}
    db_conn = db_registry[db]

    # 2. Execution accuracy
    for r in data["results"]:
        q = gold.get(r["question"])
        if r.get("error") or not q or not r.get("generated_sql"):
            continue
        if judged_only(q):
            r["exec_exact"], r["compare"] = None, "judge"
            continue
        r.pop("compare", None)
        try:
            score = execution_accuracy(db_conn, q.get("accepted_queries", []), r["generated_sql"])
        except DatabaseUnreachable as e:
            logger.error(f"Database unreachable, score kept: {r['question']} — {e}")
            continue
        exact = score["exact"] if score else None
        if exact != r.get("exec_exact"):
            logger.info(f"exec {r.get('exec_exact')} → {exact}: {r['question']}")
        r["exec_exact"] = exact

    # 3. Summary
    summary = compute_summary(data["results"])
    meta = {**data.get("meta", {}), "rescored_at": datetime.now().isoformat(timespec="minutes")}
    save_results(data["results"], summary, path, meta=meta)
    record(path, data["results"], RESCORE, at=meta["rescored_at"], commit=meta.get("commit"))
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation")
    parser.add_argument("--db", default="Chinook")
    parser.add_argument("--benchmark", help="Questions file (default: datasets/<db>.yaml)")
    parser.add_argument("--difficulty", choices=["simple", "medium", "hard"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Results file (default: results/<db>.json)")
    parser.add_argument("--rescore", action="store_true",
                        help="Compare the saved queries with the gold again, no LLM call (after a change of the comparison)")
    parser.add_argument("--only", action="append", metavar="WORD",
                        help="Run again only the questions holding this word (repeatable), merged into the saved results")
    args = parser.parse_args()
    output = Path(args.output or results_path(args.db))

    if args.rescore:
        print_summary(rescore(args.db, args.benchmark, output))
        return
    if args.only and not output.exists():
        parser.error(f"--only merges into {output}, which does not exist: run the whole eval first")

    # 1. Run evaluation
    runner = EvaluationRunner(db=args.db)
    runner.run(benchmark=args.benchmark, difficulty=args.difficulty, limit=args.limit, only=args.only)

    # 2. Display
    df = results_to_df(runner.results)
    print(df.drop(columns=["sql"]).to_string(index=False))

    # 3. Persist: a partial run replaces its questions in the saved run, which keeps its own meta
    if args.only:
        saved = json.loads(output.read_text())
        results = merge_results(saved["results"], runner.results)
        now = runner.meta()
        rerun = {"at": now["run_at"], "commit": now["commit"], "questions": [r["question"] for r in runner.results]}
        meta = {**saved.get("meta", {}), "partial_reruns": [*saved.get("meta", {}).get("partial_reruns", []), rerun]}
        summary = compute_summary(results)
        logger.info(f"{len(runner.results)} question(s) replaced in {output}")
        state = {"kind": PARTIAL, "at": rerun["at"], "commit": rerun["commit"]}
    else:
        results, summary, meta = runner.results, runner.summary, runner.meta()
        state = {"kind": RUN, "at": meta["run_at"], "commit": meta["commit"]}
    print_summary(summary)
    save_results(results, summary, output, meta=meta)
    # 4. The score of this state in the history of the evals, for the studio
    record(output, results, **state)


if __name__ == "__main__":
    main()