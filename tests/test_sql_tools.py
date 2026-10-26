"""Text values a query filters on, and whether it computes an expression (src/lib/sql_tools.py)."""

import pytest

from lib.sql_tools import contains_expression, get_value_filters


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


@pytest.mark.parametrize("query, expression, sources, contained", [
    # Two metrics (table.column): the same columns of two tables are two formulas
    ("COUNT(DISTINCT eden.numero_contrat)", "COUNT(DISTINCT wafasalaf.numero_contrat)", None, False),
    ("SUM(eden.montant)", "SUM(trade.montant)", None, False),
    ("sum( eden.Montant )", "SUM(`eden`.`montant`)", None, True),
    # A query: an alias stands for its table, an unqualified column for the only table read
    ("SELECT SUM(od.unitprice * od.quantity) FROM order_details od",
     "SUM(order_details.unitprice * order_details.quantity)", None, True),
    ("SELECT SUM(od.unitprice * od.quantity) FROM order_details od",
     "SUM(orders.unitprice * orders.quantity)", None, False),
    ("SELECT SUM(unitprice * quantity) FROM order_details", "SUM(order_details.unitprice * order_details.quantity)", None, True),
    ("SELECT SUM(`E`.montant) FROM awb.Eden e", "SUM(eden.montant)", None, True),
    ("SELECT SUM(e.montant) FROM eden e JOIN trade t ON e.id = t.id", "SUM(trade.montant)", None, False),
    # A ratio contains its parts
    ("SELECT SUM(e.montant) / COUNT(DISTINCT e.numero_contrat) FROM eden e", "SUM(eden.montant)", None, True),
    # Read from a step of the query (CTE): its table can't be told, any table matches
    ("WITH c AS (SELECT DISTINCT numero_contrat, montant FROM eden) SELECT SUM(c.montant) FROM c",
     "SUM(eden.montant)", None, True),
    # An output of the lineage (an expression alone) with the tables of its step
    ("SUM(i.total)", "SUM(invoice.total)", {"i": "invoice"}, True),
    ("SUM(i.total)", "SUM(orders.total)", {"i": "invoice"}, False),
    ("SUM(c.total)", "SUM(invoice.total)", {"c": None}, True),
    # Unparseable
    ("SELECT SUM(e.montant FROM eden e", "SUM(eden.montant)", None, False),
])
def test_contains_expression_compares_the_tables(query, expression, sources, contained):
    assert contains_expression(query, expression, "impala", sources) is contained
