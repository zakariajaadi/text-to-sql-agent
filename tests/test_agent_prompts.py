"""Agent prompts (src/agent/prompts.py): they format, and keep the rules the pilot relies on."""

import string

import pytest

from agent.prompts import DIALECT_RULES, PLANNER_PROMPT, RERANK_TABLES_PROMPT, SQL_CODER_PROMPT


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_the_prompts_format_with_their_placeholders(prompt):
    fields = {name for _, name, _, _ in string.Formatter().parse(prompt) if name}
    assert "{" not in prompt.format(**{name: "x" for name in fields})


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_no_filter_the_question_does_not_ask_for(prompt):
    # Neither on the product a table is limited to, nor to clean the data
    assert "product or offer" in prompt and "clean the data" in prompt


@pytest.mark.parametrize("dialect", ["hive", "impala"])
def test_only_iso_date_patterns_are_cast(dialect):
    # A DB2 timestamp (yyyy-MM-dd-HH.mm.ss) starts like an ISO date, but CAST turns it into NULL
    assert "When the pattern is yyyy-MM-dd or yyyy-MM-dd HH:mm:ss, use CAST" in " ".join(DIALECT_RULES[dialect].split())


@pytest.mark.parametrize("prompt", [RERANK_TABLES_PROMPT, PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_information_in_no_column_is_not_answered_with_another_one(prompt):
    # The reranker says out of scope; the planner and sql_coder write "Not available" (agent/nodes.py)
    text = " ".join(prompt.split())
    assert "in no column" in text and "an identifier, a code, a label" in text


def test_the_coder_refusal_line_is_the_one_the_nodes_read():
    from lib.sql_tools import is_refusal
    line = next(line for line in SQL_CODER_PROMPT.splitlines() if line.startswith("-- Not available"))
    assert is_refusal(line, "impala")
    assert "SELECT a text message" in " ".join(SQL_CODER_PROMPT.split())
