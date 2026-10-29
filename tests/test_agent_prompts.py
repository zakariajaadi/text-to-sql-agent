"""Agent prompts (src/agent/prompts.py): they format, and keep the rules the pilot relies on."""

import string

import pytest

from agent.prompts import (
    DIALECT_RULES,
    FORMAT_ANSWER_PROMPT,
    PLANNER_PROMPT,
    RERANK_TABLES_PROMPT,
    SQL_CODER_PROMPT,
)


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


@pytest.mark.parametrize("prompt", [RERANK_TABLES_PROMPT, PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_an_empty_table_is_not_missing_information(prompt):
    # A count over a table described as empty is 0, not "Not available (absent)"
    text = " ".join(prompt.split())
    assert "A table with no rows" in text and "a count over it is 0" in text


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_entities_are_always_counted_distinct(prompt):
    # No exception for a "One row per" header: a description can be wrong by a few duplicates
    text = " ".join(prompt.split())
    assert "always" in text and 'even when the table header says "One row per"' in text
    assert "unless the table header says" not in text and "only when the table header says" not in text


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_no_join_just_to_count_an_identifier(prompt):
    text = " ".join(prompt.split())
    assert "Join a table only to read or filter one of its columns" in text


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_a_negative_filter_keeps_the_null_rows(prompt):
    # "customers outside California" includes the ones whose state is NULL (eval: null_handling)
    assert "col IS NULL OR col <> 'X'" in " ".join(prompt.split())


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_grouped_aggregates_are_ordered_by_their_measure(prompt):
    assert "by its measure, descending" in " ".join(prompt.split())


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_a_measure_is_never_summed_through_a_fan_out_join(prompt):
    # The "(many A per B)" note of the "-- Joins:" lines (lib/semantics.py render_table)
    text = " ".join(prompt.split())
    assert '"(many A per B)"' in text and "never SUM or AVG a column of B" in text.replace("Never SUM", "never SUM")


def test_the_refusal_lines_are_the_ones_the_nodes_read():
    from lib.messages import not_available_reason
    from lib.sql_tools import is_refusal
    lines = [line for line in SQL_CODER_PROMPT.splitlines() if line.startswith("-- Not available")]
    assert [not_available_reason(line) for line in lines] == ["restricted", "absent"]
    assert all(is_refusal(line, "impala") for line in lines)
    assert "SELECT a text message" in " ".join(SQL_CODER_PROMPT.split())
    plan_lines = " ".join(PLANNER_PROMPT.split())
    assert '"Not available (restricted): [the information]"' in plan_lines
    assert '"Not available (absent): [the information]"' in plan_lines


@pytest.mark.parametrize("prompt", [PLANNER_PROMPT, SQL_CODER_PROMPT])
def test_the_missing_information_is_named_in_business_words(prompt):
    assert "never a table or column name" in " ".join(prompt.split())


def test_the_answer_tells_confidential_from_absent():
    text = " ".join(FORMAT_ANSWER_PROMPT.split())
    assert "this information is confidential, not accessible from the assistant, for data protection" in text
    assert "this information is not in the available data" in text
    assert "never say it is forbidden" in text
    assert "same language as the user's question" in text
