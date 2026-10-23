"""
Eval scoring (offline/eval/eval_lib): the execution comparison goes both ways, and a technical
error (database unreachable, agent crash) counts in no score.
"""

from unittest.mock import MagicMock

import pytest

from offline.eval.eval_lib import DatabaseUnreachable, compute_summary, execution_accuracy, uniform_score
from offline.eval.eval_lib.execution import _exact_execution_score

GOLD = [("a", 1), ("b", 2), ("c", 3)]


@pytest.mark.parametrize("gen, ordered, limited, score", [
    (GOLD, False, False, 1.0),
    ([("c", 3), ("a", 1), ("b", 2)], False, False, 1.0),            # order of rows
    ([(10, "a", 1), (20, "b", 2), (30, "c", 3)], False, False, 1.0),  # an extra column
    ([("a", 1), ("b", 2)], False, False, 0.0),                       # a row missing (unrequested filter)
    ([("a", 1), ("b", 2), ("c", 3), ("d", 4)], False, False, 0.0),   # a row too many
    ([("a", 1)], False, True, 1.0),                                  # LIMIT: one of the gold answers
    ([("z", 9)], False, True, 0.0),                                  # LIMIT: still a gold row
    (GOLD + [("d", 4)], True, False, 1.0),                           # full ranking, gold top 3 first
    ([("d", 4)] + GOLD, True, False, 0.0),                           # badly ordered ranking
])
def test_exact_execution_score(gen, ordered, limited, score):
    assert _exact_execution_score(GOLD, gen, ordered, limited) == score


def test_a_limit_in_a_comment_does_not_count(monkeypatch):
    rows = {"gold": [("a",), ("b",)], "gen": [("a",)]}
    monkeypatch.setattr("offline.eval.eval_lib.execution._execute_query",
                        lambda db, sql: rows["gold"] if sql == "gold" else rows["gen"])
    assert execution_accuracy(None, ["gold"], "-- Assumption: limit 5 shown\nSELECT x FROM t") == {"exact": 0.0}
    assert execution_accuracy(None, ["gold"], "SELECT x FROM t LIMIT 1") == {"exact": 1.0}


def test_an_unreachable_database_is_not_a_wrong_query():
    db = MagicMock()
    db._engine.connect.side_effect = ConnectionRefusedError("connection refused")
    with pytest.raises(DatabaseUnreachable):
        execution_accuracy(db, ["SELECT 1"], "SELECT 1")


def test_a_partial_run_replaces_only_its_questions():
    from offline.eval.run import merge_results
    saved = [{"question": "a", "exec_exact": 1.0}, {"question": "b", "exec_exact": 0.0}, {"question": "c", "exec_exact": 1.0}]
    new = [{"question": "b", "exec_exact": 1.0}, {"question": "d", "exec_exact": 1.0}]
    assert merge_results(saved, new) == [
        {"question": "a", "exec_exact": 1.0}, {"question": "b", "exec_exact": 1.0},
        {"question": "c", "exec_exact": 1.0}, {"question": "d", "exec_exact": 1.0},
    ]


def _result(score, exec_exact, error=None):
    r = {"question": "q", "difficulty": "simple", "category": "count", "latency_s": 0 if error else 5.0,
         "table_recall": 0 if error else 1.0, "exec_exact": exec_exact, "judge": uniform_score(score, "")}
    if error:
        r["error"] = error
    return r


def test_technical_errors_count_in_no_score():
    summary = compute_summary([_result(5, 1.0), _result(5, 1.0), _result(0, None, error="unreachable")])
    assert summary["total_questions"] == 3
    assert summary["technical_errors"] == 1
    assert summary["exec_exact"] == 1.0
    assert summary["avg_table_recall"] == 1.0
    assert summary["avg_latency_s"] == 5.0
