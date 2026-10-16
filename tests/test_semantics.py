"""Matching values written in the rendered DDL (src/lib/semantics.py), and kept by column pruning."""

from lib.column_pruning import get_column_names, prune_ddl
from lib.semantics import add_matching_values

DDL = """-- Table: customers
-- Description: One row per customer.

CREATE TABLE `customers` (
\t`customerid` STRING,  -- Unique code of the customer.
\t`city` STRING,  -- City of the company.
\t`fax` STRING
)"""


def test_values_are_appended_to_the_column_comment():
    ddl = add_matching_values(DDL, {"city": ["Sao Paulo", "São Paulo"]})
    assert "\t`city` STRING,  -- City of the company. Matching values: 'Sao Paulo', 'São Paulo'" in ddl


def test_a_column_without_comment_gets_one():
    ddl = add_matching_values(DDL, {"fax": ["030-0074321"]})
    assert "\t`fax` STRING  -- Matching values: '030-0074321'" in ddl


def test_other_lines_are_left_as_they_are():
    assert add_matching_values(DDL, {}) == DDL
    ddl = add_matching_values(DDL, {"city": ["London"]})
    assert ddl.splitlines()[:5] == DDL.splitlines()[:5]
    assert get_column_names(ddl) == ["customerid", "city", "fax"]


def test_pruning_keeps_the_matching_values():
    ddl = prune_ddl(add_matching_values(DDL, {"city": ["London"]}), {"customerid", "city"})
    assert "Matching values: 'London'" in ddl
    assert "fax" not in ddl


def test_left_out_values_are_counted():
    ddl = add_matching_values(DDL, {"city": ["Santa Clara", "Santa Cruz"]}, {"city": 3})
    assert "Matching values: 'Santa Clara', 'Santa Cruz' (+3 more)" in ddl
