# offline/eval/run.py
# ===========
# EvaluationRunner — core orchestration for Text-to-SQL benchmarking.
#
# Usage (CLI, from the project root, PYTHONPATH=src — see `make eval`):
#   python -m offline.eval.run --db Northwind                   # datasets/northwind.yaml → results/northwind.json
#   python -m offline.eval.run --difficulty hard                # filter by difficulty
#   python -m offline.eval.run --limit 5                        # run first N only
#
# Usage (importable):
#   from offline.eval.run import EvaluationRunner
#   from offline.eval.eval_lib import results_to_df, print_summary
#   runner = EvaluationRunner(db="Chinook")
#   runner.run(difficulty="hard")
#   df = results_to_df(runner.results)
#   print_summary(runner.summary)

import argparse
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
from resources import db_registry

# Inputs and outputs of the eval, one file per database (Chinook → datasets/chinook.yaml, results/chinook.json)
EVAL_DIR = Path(__file__).resolve().parent


def dataset_path(db: str) -> Path:
    return EVAL_DIR / "datasets" / f"{db.lower()}.yaml"


def results_path(db: str) -> Path:
    """Tracked by git: after a run, `git diff` shows the change against the last committed results."""
    return EVAL_DIR / "results" / f"{db.lower()}.json"


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
            difficulty: str = None, limit: int = None) -> list[dict]:
        """
        Main entry point — runs the full evaluation pipeline.
        1. Load and filter questions from YAML
        2. For each question: run agent → compute metrics → judge SQL → execution accuracy
        3. Compute aggregate summary
        """
        # 1. Load questions
        questions = load_questions(benchmark or dataset_path(self.db), difficulty)
        if limit:
            questions = questions[:limit]

        logger.info(f"Running {len(questions)} questions on '{self.db}'")
        self.results = []

        # 2. Evaluate each question
        for i, q in enumerate(questions):
            if i > 0:
                time.sleep(2)  # throttle to avoid rate limits

            logger.info(f"[{i+1}/{len(questions)}] {q['question']}")

            try:
                result = self._evaluate_one(q)
                # Retry once if agent failed to generate SQL
                if result["generated_sql"] is None and q["category"] != "out_of_scope":
                    logger.warning(f"Retrying: {q['question']}")
                    time.sleep(5)
                    result = self._evaluate_one(q)
            except Exception as e:
                logger.error(f"Agent failed: {q['question']} — {e}")
                result = {
                    "question": q["question"],
                    "difficulty": q["difficulty"],
                    "category": q["category"],
                    "generated_sql": None,
                    "selected_tables": [],
                    "latency_s": 0,
                    "table_recall": 0, "table_precision": 0, "table_hallucination": 0,
                    "exec_exact": None,
                    "judge": uniform_score(0, f"Agent error: {e}"),
                }

            self.results.append(result)

        # 3. Aggregate
        self.summary = compute_summary(self.results)
        return self.results

    def _evaluate_one(self, q: dict) -> dict:
        """
        Evaluate a single benchmark question.
        1. Run agent → get SQL + selected tables
        2. Compute table recall/precision (programmatic)
        3. Execution accuracy — compare result sets against gold queries
        4. LLM judge → 6-dimension scores
        """
        # 1. Run agent
        agent_res = self._run_agent(
            q["question"],
            tags=["eval", f"difficulty:{q['difficulty']}"]
        )

        # 2. Table metrics
        table_metrics = compute_table_metrics(
            q["expected_tables"], agent_res["selected_tables"]
        )

        # 3. Execution accuracy
        exec_score = execution_accuracy(
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
            )),
        ])

        return response.model_dump()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation")
    parser.add_argument("--db", default="Chinook")
    parser.add_argument("--benchmark", help="Questions file (default: datasets/<db>.yaml)")
    parser.add_argument("--difficulty", choices=["simple", "medium", "hard"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Results file (default: results/<db>.json)")
    args = parser.parse_args()

    # 1. Run evaluation
    runner = EvaluationRunner(db=args.db)
    runner.run(benchmark=args.benchmark, difficulty=args.difficulty, limit=args.limit)

    # 2. Display
    df = results_to_df(runner.results)
    print(df.drop(columns=["sql"]).to_string(index=False))
    print_summary(runner.summary)

    # 3. Persist
    save_results(runner.results, runner.summary, args.output or results_path(args.db), meta=runner.meta())


if __name__ == "__main__":
    main()