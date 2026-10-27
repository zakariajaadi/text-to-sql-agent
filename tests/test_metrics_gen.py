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


def test_dump_metrics_writes_what_the_agent_reads_then_the_origin():
    """origin (the generator) and edited (the studio) come after the fields the agent reads, whatever their order."""
    text = dump_metrics({"revenue": {"edited": ["expr"], "origin": "ai", "expr": "SUM(t.x)", "description": "Sold"}}, [])
    assert list(yaml.safe_load(text)["revenue"]) == ["description", "expr", "origin", "edited"]
