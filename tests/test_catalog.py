# The compact table catalog shown to the table selector (lib/catalog.py)

from lib.catalog import render_catalog, render_catalog_entry

ORDERS = """-- Table: orders
-- Description: One row per customer order.
-- Joins: orders.customerid = customers.customerid (many orders per customers)

CREATE TABLE `orders` (
\t`orderid` INT,  -- Unique order identifier.
\t`customerid` STRING,  -- Customer who placed the order.
\t`freight` DECIMAL,  -- Shipping cost.
)

/*
3 rows from orders table:
orderid\tcustomerid\tfreight
1\tALFKI\t32.38
*/"""


def test_entry_has_name_description_columns_and_joined_tables():
    entry = render_catalog_entry("orders", ORDERS, ["shippers", "customers"])
    assert entry == ("### orders\n"
                     "One row per customer order.\n"
                     "Columns: orderid, customerid, freight\n"
                     "Joins: customers, shippers")


def test_entry_without_description_or_joins_keeps_the_name_and_columns():
    entry = render_catalog_entry("t", "CREATE TABLE `t` (\n\t`a` INT,\n\t`b` INT\n)")
    assert entry == "### t\nColumns: a, b"


def test_catalog_keeps_the_given_order_and_skips_unknown_tables():
    documents = {"orders": ORDERS, "customers": "-- Table: customers\n-- Description: One row per customer.\n"}
    catalog   = render_catalog(["customers", "ghost", "orders"], documents, {"orders": ["customers"]})
    assert catalog.index("### customers") < catalog.index("### orders")
    assert "ghost" not in catalog
    assert "Joins: customers" in catalog
