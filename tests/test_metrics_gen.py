"""Metrics generator (offline/semantic_gen/metrics.py): proposals skipped before validation (utils.skip_reason)."""

import pytest

import yaml

from offline.semantic_gen.utils import dump_metrics, skip_reason

KNOWN   = {"revenue": {"expr": "SUM(invoice_line.unitprice * invoice_line.quantity)"}}
IGNORED = ["order_count"]


@pytest.mark.parametrize("name, expr, skipped", [
    # Named like a defined or rejected metric
    ("revenue", "SUM(invoice_line.quantity)", True),
    ("order_count", "COUNT(DISTINCT invoice.invoiceid)", True),
    # A known expression under another name, spelling and quoting aside
    ("sales", "sum( invoice_line.UnitPrice * `invoice_line`.quantity )", True),
    # ... but not the same columns of another table: each metric counts its own table
    ("order_sales", "SUM(order_line.unitprice * order_line.quantity)", False),
    # Distinct amounts, whatever the name and the spelling: two rows with the same amount are two amounts
    ("outstanding", "SUM(DISTINCT loan.outstanding_amount)", True),
    ("outstanding_bis", "sum( distinct loan.outstanding_amount ) / COUNT(DISTINCT loan.loan_id)", True),
    ("average_premium", "AVG(DISTINCT policy.premium)", True),
    # A plain count of one column: no business rule, and it would force its table into the questions
    ("loan_count", "COUNT(DISTINCT loan.loan_id)", True),
    ("row_count", "count(*)", True),
    ("client_count", "COUNT( DISTINCT `client`.`client_id` )", True),
    # New metrics, a count with a business rule included
    ("quantity", "SUM(invoice_line.quantity)", False),
    ("average_loan", "SUM(loan.amount) / COUNT(DISTINCT loan.client_id)", False),
    ("late_loan_count", "COUNT(DISTINCT CASE WHEN loan.status = 'LATE' THEN loan.loan_id END)", False),
])
def test_skip_reason(name, expr, skipped):
    assert (skip_reason(name, expr, KNOWN, IGNORED, "sqlite") is not None) is skipped


def test_dimension_words_are_the_vocabulary_in_use():
    from offline.semantic_gen.metrics import _dimension_words
    entries = {"a": {"dimensions": ["client", "agence"]}, "b": {"dimensions": ["client"]}, "c": {}}
    assert _dimension_words(entries) == ["agence", "client"]


def test_render_roles_lists_the_joined_tables_too():
    """A date or an identifier of a joined table is usable (prompt rules 6 and 7): its roles are rendered."""
    from offline.semantic_gen.metrics import _render_roles

    roles = {"booking":  {"measures": ["seat_price"], "identifiers": ["booking_id"], "dates": []},
             "calendar": {"measures": [], "identifiers": ["day_id"], "dates": ["flight_date"]}}
    text = _render_roles(roles, ["calendar", "booking"], fact_tables={"booking"})
    assert text.splitlines() == [
        "- booking (fact table): measures seat_price; identifiers booking_id; dates none",
        "- calendar: measures none; identifiers day_id; dates flight_date",
    ]


def test_the_metrics_prompt_shows_the_safe_forms():
    """Filtered amounts via CASE, NULLIF on ratio denominators, and a worked example (few-shot)."""
    from offline.semantic_gen.prompts import METRICS_PROMPT

    text = " ".join(METRICS_PROMPT.split())
    assert "SUM(CASE WHEN ... THEN measure END)" in text and "expr never carries a WHERE" in text
    assert "NULLIF(..., 0)" in text
    assert "## EXAMPLE" in text and "NULLIF(COUNT(DISTINCT booking.booking_id), 0)" in text


def test_a_numeric_code_is_never_a_measure(tmp_path):
    """A code stored as a number (code_agence, numero_dossier) is never offered to the LLM as a measure."""
    import sqlalchemy
    from langchain_community.utilities import SQLDatabase

    from offline.semantic_gen.metrics import _column_roles

    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'bank.db'}")
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE pret (pret_id INTEGER, montant INTEGER, code_agence INTEGER, numero_dossier INTEGER)"))
    roles = _column_roles(SQLDatabase(engine), [], {})
    assert roles["pret"]["measures"] == ["montant"]
    assert roles["pret"]["identifiers"] == ["pret_id"]
    assert set(roles["pret"]["dimensions"]) == {"code_agence", "numero_dossier"}


def test_dump_metrics_writes_what_the_agent_reads_then_the_origin():
    """origin (the generator) and edited (the studio) come after the fields the agent reads, whatever their order."""
    text = dump_metrics({"revenue": {"edited": ["expr"], "origin": "ai", "expr": "SUM(t.x)", "description": "Sold"}}, [])
    assert list(yaml.safe_load(text)["revenue"]) == ["description", "expr", "origin", "edited"]
