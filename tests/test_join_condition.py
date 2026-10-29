"""Join keys stored as text cast in the "-- Joins" lines (lib/sql_tools.py join_condition), read back by column pruning."""

import pytest

from lib.column_pruning import get_column_types, get_structural_columns
from lib.semantics import render_table
from lib.sql_tools import join_condition

TYPES = {
    "review.customerid":   "STRING",
    "customer.customerid": "INT",
    "invoice.customerid":  "DECIMAL(8, 0)",
    "claim.customerid":    "VARCHAR(20)",
    "payment.amount":      "DECIMAL(10, 2)",
    "refund.amount":       "STRING",
}


@pytest.mark.parametrize("source, target, expected", [
    # Text on either side: that side is cast to BIGINT (join keys are identifiers)
    ("review.customerid", "customer.customerid", "CAST(review.customerid AS BIGINT) = customer.customerid"),
    ("invoice.customerid", "claim.customerid", "invoice.customerid = CAST(claim.customerid AS BIGINT)"),
    # A number type with decimals is kept
    ("refund.amount", "payment.amount", "CAST(refund.amount AS DECIMAL(10, 2)) = payment.amount"),
    # Two numbers, two texts, or an unknown type: no cast
    ("invoice.customerid", "customer.customerid", "invoice.customerid = customer.customerid"),
    ("review.customerid", "claim.customerid", "review.customerid = claim.customerid"),
    ("unknown.customerid", "customer.customerid", "unknown.customerid = customer.customerid"),
])
@pytest.mark.parametrize("dialect", ["hive", "impala"])
def test_text_join_keys_are_cast(source, target, expected, dialect):
    assert join_condition(source, target, TYPES, dialect) == expected


def test_column_types_are_read_from_the_ddl():
    ddl = ("CREATE TABLE `invoice` (\n"
           "\t`invoiceid` INT,  -- Unique identifier.\n"
           "\t`customerid` DECIMAL(8, 0),\n"
           "\tbillingcity STRING\n"
           ")")
    assert get_column_types(ddl) == {"invoiceid": "INT", "customerid": "DECIMAL(8, 0)", "billingcity": "STRING"}


def test_the_joins_line_holds_the_cast_and_pruning_keeps_the_key():
    ddl = "CREATE TABLE `review` (\n\t`reviewid` INT,\n\t`customerid` STRING,\n\t`rating` INT\n)"
    doc = render_table("review", ddl, {"description": "One row per review."},
                       [("review.customerid", "customer.customerid")],
                       column_types=TYPES, dialect="impala")

    # The cardinality note (from = N side): a measure of customer is never summed through this join
    assert "-- Joins: CAST(review.customerid AS BIGINT) = customer.customerid (many review per customer)" in doc
    assert get_structural_columns("review", doc, query_tables={"review", "customer"}) == {"customerid"}
    # Read back from the other side too, the cast at the end of a decimal type
    line = "-- Joins: invoice.customerid = CAST(claim.customerid AS DECIMAL(10, 2))"
    claim = f"{line}\nCREATE TABLE `claim` (\n\t`customerid` VARCHAR(20)\n)"
    assert get_structural_columns("claim", claim, query_tables={"invoice", "claim"}) == {"customerid"}


def test_pruning_keeps_the_identifier_other_tables_point_to_even_alone():
    # Counting people on the person table alone needs its id (COUNT(DISTINCT id))
    doc = ("-- Joins: claim.person_id = person.person_id, person.agency_id = agency.agency_id\n"
           "CREATE TABLE `person` (\n\t`person_id` BIGINT,\n\t`agency_id` INT,\n\t`sex` STRING\n)")
    assert get_structural_columns("person", doc, query_tables={"person"}) == {"person_id"}
    assert get_structural_columns("person", doc, query_tables={"person", "agency"}) == {"person_id", "agency_id"}
