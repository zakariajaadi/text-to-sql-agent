# src/lib/column_pruning.py

"""
Column pruning helpers — deterministic text operations on rendered DDL documents.
Used by: agent/nodes.py (prune_columns)

A rendered DDL is the per-table document indexed in Chroma by
render_table (lib/semantics.py):

    -- Table: customer_cdc
    -- Description: One row per change event captured for a customer record...
    -- Joins: customer_cdc.customerid = customer.customerid
    -- Read mode: LIVE (key: customerid, version: time)

    CREATE TABLE `customer_cdc` (
        `customerid` INT,  -- Unique identifier for the customer.
        `email` STRING,  -- Email address.
        ...
    )

    /*
    3 rows from customer_cdc table:
    customerid	email	...
    1	luisg@embraer.com.br	...
    */

Two kinds of columns:
    - semantic columns   — depend on the question (email, total...): chosen by the LLM.
    - structural columns — needed by the query mechanics whatever the question
                           (join keys, partition / key / version columns): never pruned.
"""

import re


def _parse_column_name(line: str) -> str:
    """Column name of a CREATE TABLE body line, unquoted: "`email` STRING, -- ..." → "email"."""
    return line.strip().split()[0].strip('`"[]')


def _create_body_lines(ddl: str) -> list[str]:
    """The column lines of the CREATE TABLE body of a DDL, in declaration order."""
    lines          = []
    in_create_body = False

    for line in ddl.splitlines():
        if line.startswith("CREATE TABLE"):
            in_create_body = True
        elif line.startswith(")"):
            in_create_body = False
        elif in_create_body and line.strip():
            lines.append(line)

    return lines


def get_column_names(ddl: str) -> list[str]:
    """Column names declared in the CREATE TABLE body of a rendered DDL, in declaration order."""
    return [_parse_column_name(line) for line in _create_body_lines(ddl)]


def get_column_types(ddl: str) -> dict[str, str]:
    """Type of each column of a DDL: "`amount` DECIMAL(8, 0),  -- ..." → {"amount": "DECIMAL(8, 0)"}."""
    types = {}
    for line in _create_body_lines(ddl):
        match = re.match(r"\s*\S+\s+(\w+(?:\s*\([^)]*\))?)", line)
        if match:
            types[_parse_column_name(line)] = match.group(1)
    return types


def get_structural_columns(table_name: str, ddl: str, query_tables: set[str]) -> set[str]:
    """
    Columns the generated SQL needs whatever the question asks — never pruned.

    Read from the header lines written by render_table (lib/semantics.py):
        - "-- Joins:"     → this table's join keys towards the other tables of the query
                            (a key towards a table absent from the query cannot be used),
                            and its own identifier, the key other tables point to (right side
                            of "source = target"), whatever the query: entities are counted on it
        - "-- Read mode:" → partition column (LAST, ALL), key and version columns (LIVE)

    Without them, the query can neither join the selected tables, count the table's
    entities (COUNT(DISTINCT id)) nor filter the right partition / latest version.
    """
    candidate_columns = set()

    for line in ddl.splitlines():
        if line.startswith("-- Joins:"):
            # "invoice.customerid = customer.customerid" → keep this table's side
            # when the other side is in the query (self-joins keep both sides).
            # A side may be cast: "CAST(review.customerid AS BIGINT) = customer.customerid"
            for left_table, left_column, right_table, right_column in re.findall(
                r"(\w+)\.(\w+)(?:\s+AS\s+\w+(?:\([\d,\s]*\))?\))?\s*=\s*(?:CAST\()?(\w+)\.(\w+)", line, re.IGNORECASE
            ):
                if left_table == table_name and right_table in query_tables:
                    candidate_columns.add(left_column)
                if right_table == table_name:   # this table's own identifier, kept even alone
                    candidate_columns.add(right_column)
        elif line.startswith("-- Read mode:"):
            # "LIVE (key: customerid, version: time)" → every word inside the parentheses;
            # the labels (key, version, partition) are dropped by the intersection below
            candidate_columns |= set(re.findall(r"\w+", line.partition("(")[2]))

    # Keep only names that are real columns of the table
    return candidate_columns & set(get_column_names(ddl))


def prune_ddl(ddl: str, columns_to_keep: set[str]) -> str:
    """
    Remove from a rendered DDL every column not in `columns_to_keep`.

    Two sections are pruned, the rest (header comments, CREATE TABLE line) is kept as is:
        - CREATE TABLE body  — one line per column, dropped when not kept.
        - sample rows block  — tab-separated values, pruned by column position.

    A trailing comma may remain on the last kept column: harmless,
    the DDL is only read by the LLM, never executed.
    """
    pruned_lines   = []
    section        = None   # "create_body", "sample_rows" or None
    kept_positions = None   # positions of kept columns in the sample rows, set from their header

    for line in ddl.splitlines():
        if line.startswith("CREATE TABLE"):
            section = "create_body"
        elif line.startswith("/*"):
            section = "sample_rows"
        elif line.startswith(")") or line.startswith("*/"):
            section = None
        elif section == "create_body" and line.strip():
            if _parse_column_name(line) not in columns_to_keep:
                continue
        elif section == "sample_rows" and "\t" in line:
            cells = line.split("\t")
            # First tab-separated line is the header: column names give the positions to keep
            if kept_positions is None:
                kept_positions = [i for i, cell in enumerate(cells) if cell in columns_to_keep]
            line = "\t".join(cells[i] for i in kept_positions)

        pruned_lines.append(line)

    return "\n".join(pruned_lines)