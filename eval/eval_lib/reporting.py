# eval/eval_lib/reporting.py
# =====================
# Output functions — DataFrame conversion, summary stats, JSON export.
# Transforms raw results into human-readable or machine-readable formats.

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from judge_schemas import DIMENSIONS, SCORE_LABELS


def results_to_df(results: list[dict]) -> pd.DataFrame:
    """Flatten a list of evaluation results into a DataFrame for analysis."""
    rows = []
    for r in results:
        j = r["judge"]
        rows.append({
            "question": r["question"],
            "difficulty": r["difficulty"],
            "category": r["category"],
            "tbl": j["table_selection"]["score"],
            "col": j["column_selection"]["score"],
            "join": j["join_correctness"]["score"],
            "filt": j["filter_logic"]["score"],
            "agg": j["aggregation"]["score"],
            "overall": j["overall"]["score"],
            "exec": r.get("execution_accuracy"),
            "recall": r.get("table_recall", 0),
            "latency": r["latency_s"],
            "sql": r["generated_sql"] or "",
        })
    return pd.DataFrame(rows)


def compute_summary(results: list[dict]) -> dict:
    """
    Aggregate metrics across all evaluation results.
    1. Average score per judge dimension
    2. Average table recall
    3. Execution accuracy (excludes OOS/errors where score is None)
    4. Score distribution (how many 5s, 4s, etc.)
    5. Breakdown by difficulty level
    """
    if not results:
        return {}

    valid = [r for r in results if r["judge"]["overall"]["score"] > 0]

    summary = {
        "total_questions": len(results),
        "evaluated": len(valid),
        "avg_latency_s": round(
            sum(r["latency_s"] for r in results) / len(results), 2
        ),
    }

    # 1. Average scores per judge dimension
    for dim in DIMENSIONS:
        scores = [r["judge"][dim]["score"] for r in valid]
        summary[f"avg_{dim}"] = round(sum(scores) / len(scores), 2) if scores else 0

    # 2. Average table recall
    recalls = [r["table_recall"] for r in results if "table_recall" in r]
    summary["avg_table_recall"] = round(sum(recalls) / len(recalls), 2) if recalls else 0

    # 3. Execution accuracy — only count scorable questions (not None)
    exec_scores = [
        r["execution_accuracy"] for r in results
        if r.get("execution_accuracy") is not None
    ]
    summary["execution_accuracy"] = (
        round(sum(exec_scores) / len(exec_scores), 3) if exec_scores else 0
    )
    summary["execution_evaluated"] = len(exec_scores)

    # 4. Score distribution
    overall_scores = [r["judge"]["overall"]["score"] for r in valid]
    summary["score_distribution"] = {
        f"{SCORE_LABELS[s]} ({s})": overall_scores.count(s)
        for s in range(5, 0, -1)
    }

    # 5. Breakdown by difficulty
    for diff in ("simple", "medium", "hard"):
        diff_results = [r for r in valid if r["difficulty"] == diff]
        if diff_results:
            diff_scores = [r["judge"]["overall"]["score"] for r in diff_results]
            summary[f"avg_overall_{diff}"] = round(
                sum(diff_scores) / len(diff_scores), 2
            )
            # Execution accuracy per difficulty
            diff_exec = [
                r["execution_accuracy"] for r in diff_results
                if r.get("execution_accuracy") is not None
            ]
            if diff_exec:
                summary[f"exec_accuracy_{diff}"] = round(
                    sum(diff_exec) / len(diff_exec), 3
                )

    return summary


def print_summary(summary: dict):
    """Print aggregate metrics to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  Questions:         {summary['total_questions']}")
    print(f"  Avg latency:       {summary['avg_latency_s']}s")
    print(f"  Avg table recall:  {summary['avg_table_recall']}")
    print(f"  Avg overall score: {summary['avg_overall']}/5")
    print(f"  Execution accuracy: {summary['execution_accuracy']:.1%}"
          f" ({summary['execution_evaluated']} scored)")

    if "score_distribution" in summary:
        print("\n  Score distribution:")
        for label, count in summary["score_distribution"].items():
            print(f"    {label:<14} {count:>3}  {'█' * count}")

    for diff in ("simple", "medium", "hard"):
        key = f"avg_overall_{diff}"
        exec_key = f"exec_accuracy_{diff}"
        if key in summary:
            exec_str = f" | exec {summary[exec_key]:.0%}" if exec_key in summary else ""
            print(f"  Avg {diff:<8}:     {summary[key]}/5{exec_str}")

    print(f"{'=' * 50}")


def save_results(results: list[dict], summary: dict, path: str = "eval/results.json"):
    """Save results and summary to JSON for comparison."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"Results saved to {output_path}")