"""Text values a query filters on (src/lib/sql_tools.py get_value_filters)."""

import pytest

from lib.sql_tools import get_value_filters


def filters(sql):
    return get_value_filters(sql, "hive")


@pytest.mark.parametrize("sql, expected", [
    # Aliases resolved to their table, backticks or not
    ("SELECT COUNT(*) FROM customers AS c WHERE c.city = 'Sao Paolo'",
     [("customers", "city", "Sao Paolo")]),
    ("SELECT COUNT(*) FROM `customers` AS `c` WHERE `c`.`city` = 'London'",
     [("customers", "city", "London")]),
    # Table name as qualifier, value on the left
    ("SELECT COUNT(*) FROM customers WHERE 'UK' = customers.country",
     [("customers", "country", "UK")]),
    # IN, NOT IN and <>
    ("SELECT 1 FROM orders o WHERE o.shipcity IN ('Bruxelles', 'Bern') AND o.shipcountry <> 'USA'",
     [("orders", "shipcity", "Bruxelles"), ("orders", "shipcity", "Bern"), ("orders", "shipcountry", "USA")]),
    ("SELECT 1 FROM orders o WHERE o.shipcity NOT IN ('Bern')",
     [("orders", "shipcity", "Bern")]),
    # Unqualified column: no table
    ("SELECT 1 FROM customers WHERE city = 'London'",
     [("", "city", "London")]),
    # Join and subquery
    ("SELECT 1 FROM orders o JOIN customers c ON o.customerid = c.customerid "
     "WHERE c.country = 'Mexico' AND o.orderid IN (SELECT d.orderid FROM order_details d WHERE d.productid = 3)",
     [("customers", "country", "Mexico")]),
])
def test_text_filters_are_found(sql, expected):
    assert filters(sql) == expected


@pytest.mark.parametrize("sql", [
    "SELECT 1 FROM customers c WHERE LOWER(c.city) = 'london'",      # expression, not a column
    "SELECT 1 FROM customers c WHERE c.city LIKE '%Lon%'",            # LIKE
    "SELECT 1 FROM orders o WHERE o.employeeid = 3",                  # number
    "SELECT 1 FROM orders o JOIN customers c ON o.customerid = c.customerid",   # column = column
    "SELECT 1 FROM customers WHERE city = 'London",                   # unparseable
])
def test_other_conditions_are_left_out(sql):
    assert filters(sql) == []
