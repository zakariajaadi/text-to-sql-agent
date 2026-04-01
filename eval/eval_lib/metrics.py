# eval/eval_lib/metrics.py
# ===================
# Pure functions — Metrics logic.

import re

from judge_schemas import DIMENSIONS


def compute_table_metrics(expected: list[str], selected: list[str]) -> dict:
    """
    Compute table selection quality.
    - recall:        |expected ∩ selected| / |expected|
    - precision:     |expected ∩ selected| / |selected|
    - hallucination: |selected - expected| / |selected|
    """
    if not expected:
        is_empty = not selected
        return {
            "table_recall": 1.0 if is_empty else 0.0,
            "table_precision": 1.0 if is_empty else 0.0,
            "table_hallucination": 0.0 if is_empty else 1.0,
        }

    expected_set = {t.lower() for t in expected}
    selected_set = {t.lower() for t in selected}
    intersection = expected_set & selected_set

    return {
        "table_recall": round(len(intersection) / len(expected_set), 3),
        "table_precision": round(len(intersection) / len(selected_set), 3) if selected_set else 0,
        "table_hallucination": round(len(selected_set - expected_set) / len(selected_set), 3) if selected_set else 0,
    }

def uniform_score(score: int, reasoning: str) -> dict:
    """Build a uniform score across all 6 judge dimensions.
    Used for deterministic edge cases (out-of-scope, refusal)."""
    dim = {"score": score, "reasoning": reasoning}
    return {d: dim for d in DIMENSIONS}

def extract_tables(schema_text: str) -> list[str]:
    """Parse table names from CREATE TABLE statements in DDL."""
    return re.findall(r"CREATE TABLE (?:\"|')?(\w+)", schema_text)