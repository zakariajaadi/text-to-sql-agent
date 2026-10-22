"""Lineage of a query: steps, sources, joins, filters and the origins of its outputs (src/lib/query_lineage.py)."""

import pytest

from lib.query_lineage import build_query_lineage

SCHEMA = {
    "Categories":    ["CategoryID", "CategoryName"],
    "Products":      ["ProductID", "ProductName", "CategoryID", "UnitPrice"],
    "Order Details": ["OrderID", "ProductID", "UnitPrice", "Quantity", "Discount"],
    "Orders":        ["OrderID", "CustomerID", "OrderDate", "ShipCountry", "ShipVia"],
    "Shippers":      ["ShipperID", "CompanyName"],
    "Customers":     ["CustomerID", "CompanyName", "Country"],
}


def steps(sql, dialect="sqlite", columns=None):
    return {step["id"]: step for step in build_query_lineage(sql, dialect, columns)["steps"]}


def origins(step, name):
    output = next(o for o in step["outputs"] if o["name"] == name)
    return [f"{o['table']}.{o['column']}" for o in output["origins"]]


def test_joins_filter_and_aggregate():
    result = steps("""
        SELECT c.CategoryName, ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS Revenue
        FROM Categories c
        JOIN Products p ON c.CategoryID = p.CategoryID
        JOIN [Order Details] od ON p.ProductID = od.ProductID
        JOIN Orders o ON o.OrderID = od.OrderID
        WHERE strftime('%Y', o.OrderDate) = '2022' AND od.Quantity > 0
        GROUP BY c.CategoryName ORDER BY Revenue DESC LIMIT 5""")["result"]

    sources = {s["alias"]: s for s in result["sources"]}
    assert [s["name"] for s in result["sources"]] == ["Categories", "Products", "Order Details", "Orders"]
    assert sources["o"]["columns"] == {"OrderID": ["join"], "OrderDate": ["where"]}
    assert sources["c"]["columns"]["CategoryName"] == ["select", "group_by"]
    assert sources["od"]["columns"]["Quantity"] == ["select", "where"]
    assert [(j["type"], j["condition"], j["sources"]) for j in result["joins"]] == [
        ("INNER", "c.CategoryID = p.CategoryID", ["p", "c"]),
        ("INNER", "p.ProductID = od.ProductID", ["od", "p"]),
        ("INNER", "o.OrderID = od.OrderID", ["o", "od"]),
    ]
    assert [f["condition"] for f in result["filters"]] == ["STRFTIME('%Y', o.OrderDate) = '2022'", "od.Quantity > 0"]
    assert result["group_by"] == ["c.CategoryName"]
    assert result["order_by"] == ["Revenue DESC"]
    assert result["limit"] == "5"
    assert origins(result, "CategoryName") == ["Categories.CategoryName"]
    assert origins(result, "Revenue") == ["Order Details.UnitPrice", "Order Details.Quantity", "Order Details.Discount"]
    revenue = result["outputs"][1]
    assert revenue["aggregated"] and not revenue["window"]
    assert revenue["reads"] == [{"source": "od", "column": c} for c in ("UnitPrice", "Quantity", "Discount")]


def test_origins_go_through_a_cte():
    result = steps("""
        WITH OrderTotals AS (
            SELECT o.OrderID, o.ShipCountry, SUM(od.UnitPrice * od.Quantity) AS OrderTotal
            FROM Orders o JOIN [Order Details] od ON o.OrderID = od.OrderID
            GROUP BY o.OrderID, o.ShipCountry)
        SELECT ShipCountry, AVG(OrderTotal) AS AvgOrderValue FROM OrderTotals GROUP BY ShipCountry""")

    assert list(result) == ["OrderTotals", "result"]
    assert result["OrderTotals"]["kind"] == "cte"
    final = result["result"]
    assert [(s["name"], s["kind"]) for s in final["sources"]] == [("OrderTotals", "step")]
    assert final["sources"][0]["columns"] == {"ShipCountry": ["select", "group_by"], "OrderTotal": ["select"]}
    assert origins(final, "ShipCountry") == ["Orders.ShipCountry"]
    assert origins(final, "AvgOrderValue") == ["Order Details.UnitPrice", "Order Details.Quantity"]
    # reads: the columns of its own sources, here the CTE
    assert final["outputs"][1]["reads"] == [{"source": "OrderTotals", "column": "OrderTotal"}]


def test_count_star_counts_the_rows_of_its_tables():
    result = steps("""
        SELECT AVG(TrackCount) AS avg_tracks
        FROM (SELECT AlbumId, COUNT(*) AS TrackCount FROM Track GROUP BY AlbumId) AS album_tracks""")
    assert result["album_tracks"]["kind"] == "derived"
    assert origins(result["album_tracks"], "TrackCount") == ["Track.*"]
    assert result["album_tracks"]["outputs"][1]["reads"] == [{"source": "Track", "column": "*"}]
    assert origins(result["result"], "avg_tracks") == ["Track.*"]


def test_subquery_in_where_is_a_step_of_its_filter():
    result = steps("""
        SELECT c.code_agence, COUNT(DISTINCT c.numero_client_host) AS nb_clients
        FROM awb_silver.client c
        JOIN awb_silver.client_master m ON m.numero_client_host = c.numero_client_host
        WHERE m.numero_personne_host IN (SELECT numero_personne_host FROM awb_silver.impaye)""", "impala")

    assert result["subquery_1"]["kind"] == "subquery"
    assert origins(result["subquery_1"], "numero_personne_host") == ["impaye.numero_personne_host"]
    final = result["result"]
    assert [s["name"] for s in final["sources"]] == ["client", "client_master"]
    assert final["filters"][0]["steps"] == ["subquery_1"]
    assert origins(final, "nb_clients") == ["client.numero_client_host"]


def test_correlated_column_is_listed_by_the_outer_query():
    result = steps("""
        SELECT p.ProductName FROM Products p
        WHERE p.UnitPrice > (SELECT AVG(p2.UnitPrice) FROM Products p2 WHERE p2.CategoryID = p.CategoryID)""")
    outer = result["result"]["sources"][0]["columns"]
    assert outer == {"ProductName": ["select"], "UnitPrice": ["where"], "CategoryID": ["where"]}
    assert result["subquery_1"]["sources"][0]["columns"] == {"UnitPrice": ["select"], "CategoryID": ["where"]}
    # The correlation is a filter of the subquery, not a join
    assert result["subquery_1"]["joins"] == []
    assert [f["condition"] for f in result["subquery_1"]["filters"]] == ["p2.CategoryID = p.CategoryID"]


def test_comma_join_condition_moves_from_where_to_joins():
    result = steps("""
        SELECT s.CompanyName, COUNT(o.OrderID) AS n FROM Shippers s, Orders o
        WHERE s.ShipperID = o.ShipVia AND o.ShipCountry = 'France' GROUP BY s.CompanyName""")["result"]
    assert result["joins"] == [{"type": "INNER", "condition": "s.ShipperID = o.ShipVia", "sources": ["s", "o"],
                                "tables": ["Shippers", "Orders"], "keys": [["ShipperID", "ShipVia"]],
                                "pairs": [[{"source": "s", "column": "ShipperID"}, {"source": "o", "column": "ShipVia"}]],
                                "implicit": True}]
    assert [f["condition"] for f in result["filters"]] == ["o.ShipCountry = 'France'"]


def test_outer_join_and_using():
    result = steps("SELECT c.CompanyName, o.OrderDate FROM Customers c LEFT JOIN Orders o USING (CustomerID)")["result"]
    assert result["joins"] == [{"type": "LEFT", "condition": "USING (CustomerID)", "sources": ["o"], "tables": ["Orders"],
                                "keys": [["CustomerID", "CustomerID"]],
                                "pairs": [[{"source": "o", "column": "CustomerID"}, {"source": "c", "column": "CustomerID"}]],
                                "implicit": False}]


def test_unqualified_columns_need_the_schema_when_several_tables_are_read():
    sql = ("SELECT CategoryName, COUNT(ProductID) AS n FROM Categories "
           "JOIN Products USING (CategoryID) GROUP BY CategoryName ORDER BY n DESC")
    without = steps(sql)["result"]
    assert origins(without, "CategoryName") == []
    with_schema = steps(sql, columns=SCHEMA)["result"]
    assert origins(with_schema, "CategoryName") == ["Categories.CategoryName"]
    assert origins(with_schema, "n") == ["Products.ProductID"]
    # ORDER BY n names an output, not a column of a table
    assert all("n" not in s["columns"] for s in with_schema["sources"])


def test_single_table_resolves_unqualified_columns_without_schema():
    result = steps("SELECT Country, COUNT(CustomerID) AS n FROM Customers GROUP BY Country")["result"]
    assert result["sources"][0]["columns"] == {"Country": ["select", "group_by"], "CustomerID": ["select"]}
    assert origins(result, "n") == ["Customers.CustomerID"]


def test_schema_writes_the_column_names():
    result = steps("SELECT country FROM customers", columns=SCHEMA)["result"]
    assert origins(result, "country") == ["customers.Country"]


def test_select_star_is_expanded_with_the_schema():
    sql = "WITH w AS (SELECT * FROM Shippers) SELECT * FROM w"
    assert [o["name"] for o in steps(sql)["result"]["outputs"]] == ["*"]
    result = steps(sql, columns=SCHEMA)["result"]
    assert [(o["name"], origins(result, o["name"])) for o in result["outputs"]] == [
        ("ShipperID", ["Shippers.ShipperID"]), ("CompanyName", ["Shippers.CompanyName"])]


def test_window_function():
    result = steps("SELECT CustomerID, RANK() OVER (ORDER BY SUM(Freight) DESC) AS r FROM Orders GROUP BY CustomerID")["result"]
    rank = result["outputs"][1]
    assert rank["window"] and rank["aggregated"]
    assert origins(result, "r") == ["Orders.Freight"]
    running = steps("SELECT OrderID, SUM(Freight) OVER (ORDER BY OrderDate) AS running FROM Orders")["result"]["outputs"][1]
    assert running["window"] and not running["aggregated"]


def test_union_branches():
    result = steps("SELECT CompanyName FROM Customers UNION ALL SELECT CompanyName FROM Shippers ORDER BY 1")
    assert [s["kind"] for s in result.values()] == ["branch", "branch", "result"]
    final = result["result"]
    assert final["operation"] == "UNION ALL"
    assert [s["name"] for s in final["sources"]] == ["branch_1", "branch_2"]
    assert origins(final, "CompanyName") == ["Customers.CompanyName", "Shippers.CompanyName"]
    assert final["outputs"][0]["reads"] == [{"source": "branch_1", "column": "CompanyName"},
                                            {"source": "branch_2", "column": "CompanyName"}]


def test_clauses_are_written_without_needless_quotes():
    result = steps("SELECT `i`.`billingcountry` AS `pays`, COUNT(*) AS `n` FROM `invoice` AS `i` "
                   "WHERE `i`.`time` > 3 GROUP BY `i`.`billingcountry` ORDER BY `n` DESC", "hive")["result"]
    assert [f["condition"] for f in result["filters"]] == ["i.`time` > 3"]   # TIME is a keyword
    assert result["group_by"] == ["i.billingcountry"]
    assert result["order_by"] == ["n DESC"]


# ── Summaries ────────────────────────────────────────────────────────────────

def summary(sql, dialect="hive"):
    return steps(sql, dialect)["result"]["summary"]


def test_month_report_summary():
    result = steps("""
        SELECT DATE_FORMAT(i.invoicedate, 'yyyy-MM') AS mois, SUM(il.unitprice * il.quantity) AS chiffre_affaires
        FROM invoice i JOIN invoiceline il ON i.invoiceid = il.invoiceid
        WHERE i.invoicedate >= '2023-01-01 00:00:00' AND i.invoicedate <= '2023-12-31 23:59:59'
        GROUP BY DATE_FORMAT(i.invoicedate, 'yyyy-MM') ORDER BY DATE_FORMAT(i.invoicedate, 'yyyy-MM') NULLS LAST""", "hive")["result"]
    assert result["joins"][0]["tables"] == ["invoiceline", "invoice"]
    assert result["joins"][0]["keys"] == [["invoiceid", "invoiceid"]]
    [period] = result["summary"]["filters"]
    assert (period["kind"], period["column"], period["grain"], period["value"]) == ("period", "invoicedate", "year", "2023")
    assert (period["low"], period["high"]) == ("2023-01-01", "2023-12-31")
    assert result["summary"]["group_by"] == [{"column": "invoicedate", "grain": "month", "sql": "DATE_FORMAT(invoicedate, 'yyyy-MM')"}]
    assert result["summary"]["order_by"] == [{"label": "mois", "descending": False}]
    mois, ca = result["outputs"]
    assert mois["grain"] == {"grain": "month", "column": "invoicedate"}
    assert ca["formula"] == "SUM(unitprice × quantity)" and ca["grain"] is None


@pytest.mark.parametrize("sql, dialect, expected", [
    # Year written as a function or a format
    ("SELECT 1 FROM orders o WHERE strftime('%Y', o.OrderDate) = '2022'", "sqlite",
     {"kind": "period", "column": "OrderDate", "grain": "year", "value": "2022"}),
    ("SELECT 1 FROM orders WHERE YEAR(orderdate) = 2022", "hive",
     {"kind": "period", "column": "orderdate", "grain": "year", "value": "2022"}),
    ("SELECT 1 FROM orders WHERE DATE_FORMAT(orderdate, 'yyyy-MM') = '2022-03'", "hive",
     {"kind": "period", "column": "orderdate", "grain": "month", "value": "2022-03"}),
    # Upper bound excluded: the next month's first day
    ("SELECT 1 FROM orders WHERE orderdate >= '2022-02-01' AND orderdate < '2022-03-01'", "hive",
     {"kind": "period", "column": "orderdate", "grain": "month", "value": "2022-02", "last": "2022-02-28"}),
    # Impala string date parsed with its format, value on the left
    ("SELECT 1 FROM client WHERE '2020-01-01' <= TO_TIMESTAMP(date_entree, 'dd/MM/yyyy')", "impala",
     {"kind": "compare", "column": "date_entree", "grain": None, "op": ">=", "value": "2020-01-01"}),
    ("SELECT 1 FROM orders WHERE orderdate BETWEEN '2022-01-15' AND '2022-06-30'", "hive",
     {"kind": "range", "column": "orderdate", "low": "2022-01-15", "high": "2022-06-30"}),
    ("SELECT 1 FROM orders WHERE YEAR(orderdate) > 2020", "hive",
     {"kind": "compare", "column": "orderdate", "grain": "year", "op": ">", "value": "2020"}),
    ("SELECT 1 FROM orders WHERE NOT shipcountry = 'France'", "hive",
     {"kind": "compare", "column": "shipcountry", "op": "<>", "value": "France"}),
    ("SELECT 1 FROM orders WHERE shipcountry NOT IN ('France', 'Spain')", "hive",
     {"kind": "in", "column": "shipcountry", "values": ["France", "Spain"], "negated": True}),
    ("SELECT 1 FROM customers WHERE companyname LIKE 'Ana%'", "hive",
     {"kind": "like", "column": "companyname", "mode": "starts", "value": "Ana"}),
    ("SELECT 1 FROM customers WHERE companyname LIKE '%market%'", "hive",
     {"kind": "like", "column": "companyname", "mode": "contains", "value": "market"}),
    ("SELECT 1 FROM customers WHERE region IS NOT NULL", "hive",
     {"kind": "null", "column": "region", "negated": True}),
    ("SELECT 1 FROM client c WHERE c.id IN (SELECT id FROM impaye)", "hive",
     {"kind": "subquery", "column": "id", "steps": ["subquery_1"], "negated": False}),
    ("SELECT 1 FROM orders WHERE freight * 2 > unitprice", "hive",
     {"kind": "sql", "sql": "freight × 2 > unitprice"}),
])
def test_filter_summaries(sql, dialect, expected):
    [item] = summary(sql, dialect)["filters"]
    assert {k: item[k] for k in expected} == expected


@pytest.mark.parametrize("expression, expected", [
    ("YEAR(d)", ("year", "d")),
    ("QUARTER(d)", ("quarter", "d")),
    ("MONTH(d)", ("month_of_year", "d")),
    ("DATE_TRUNC('month', d)", ("month", "d")),
    ("TRUNC(d, 'MM')", ("month", "d")),
    ("SUBSTR(d, 1, 7)", ("month", "d")),
    ("CONCAT(YEAR(d), '-', MONTH(d))", ("month", "d")),
    ("DATE_FORMAT(TO_TIMESTAMP(d, 'dd/MM/yyyy'), 'yyyy-MM-dd')", ("day", "d")),
    ("UPPER(d)", None),
])
def test_date_grains(expression, expected):
    [output] = steps(f"SELECT {expression} AS x FROM t", "impala")["result"]["outputs"]
    assert (output["grain"] and (output["grain"]["grain"], output["grain"]["column"])) == (expected or None)


def test_date_format_unknown_to_the_dialect():
    [output] = steps("SELECT DATE_FORMAT(OrderDate, 'MM') AS m FROM Orders", "sqlite")["result"]["outputs"]
    assert output["grain"] == {"grain": "month_of_year", "column": "OrderDate"}


def test_order_by_position_and_group_by_position():
    result = summary("SELECT country, COUNT(*) AS n FROM customers GROUP BY 1 ORDER BY 2 DESC")
    assert result["group_by"] == [{"column": "country", "grain": None, "sql": "country"}]
    assert result["order_by"] == [{"label": "n", "descending": True}]


# ── What the graph tells: aggregations, join keys, the table the rows come from ──

@pytest.mark.parametrize("projection, expected", [
    ("ROUND(SUM(od.unitprice * od.quantity * (1 - od.discount)), 2)",
     {"function": "sum", "distinct": False, "operand": "unitprice × quantity × (1 - discount)", "round": 2}),
    ("COUNT(DISTINCT od.orderid)", {"function": "count", "distinct": True, "operand": "orderid", "round": None}),
    ("COUNT(*)", {"function": "count", "distinct": False, "operand": "*", "round": None}),
    ("ROUND(AVG(CAST(od.unitprice AS DOUBLE)))", {"function": "avg", "distinct": False, "operand": "CAST(unitprice AS DOUBLE)", "round": 0}),
    ("MAX(od.quantity)", {"function": "max", "distinct": False, "operand": "quantity", "round": None}),
    # A ratio of two aggregations is told by its formula
    ("SUM(od.unitprice) / COUNT(DISTINCT od.orderid)", None),
])
def test_aggregate_of_an_output(projection, expected):
    [output] = steps(f"SELECT {projection} AS x FROM order_details od", "hive")["result"]["outputs"]
    assert output["aggregate"] == expected


def test_window_and_plain_outputs_have_no_aggregate():
    result = steps("SELECT orderid, SUM(freight) OVER (ORDER BY orderdate) AS running FROM orders", "hive")["result"]
    assert [o["aggregate"] for o in result["outputs"]] == [None, None]


def test_join_pairs_name_their_source():
    result = steps("""
        SELECT c.categoryname, SUM(od.quantity) AS q
        FROM categories c JOIN products p ON c.categoryid = p.categoryid JOIN order_details od ON p.productid = od.productid
        GROUP BY c.categoryname""", "hive")["result"]
    assert [j["pairs"] for j in result["joins"]] == [
        [[{"source": "c", "column": "categoryid"}, {"source": "p", "column": "categoryid"}]],
        [[{"source": "p", "column": "productid"}, {"source": "od", "column": "productid"}]],
    ]


@pytest.mark.parametrize("sql, base", [
    # The table the aggregations read: its rows are the ones summed
    ("SELECT c.categoryname, SUM(od.quantity) AS q FROM categories c JOIN products p ON c.categoryid = p.categoryid "
     "JOIN order_details od ON p.productid = od.productid GROUP BY c.categoryname", "od"),
    # No aggregation: the table of the FROM
    ("SELECT c.companyname, o.orderdate FROM customers c JOIN orders o ON c.customerid = o.customerid", "c"),
    # COUNT(*) counts the rows of every source: the first one
    ("SELECT COUNT(*) AS n FROM orders o JOIN customers c ON o.customerid = c.customerid", "o"),
])
def test_base_table(sql, base):
    assert summary(sql)["base"] == base


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders WHERE name = 'unclosed",   # unparseable
    "-- Not available: the database has no salary",  # refusal, no query
])
def test_no_lineage(sql):
    assert build_query_lineage(sql, "hive") is None
