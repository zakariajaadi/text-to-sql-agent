# src/lib/value_index.py

"""
Value index — the distinct values of the categorical columns, searched at query time.
Built by: offline/indexer/schema_indexer.py (make index). Used by: agent/nodes.py (match_values, sql_validator)

The values of a categorical column are read from the database at index time
(offline/indexer/categorical_values.py). Short lists are also inlined in the table documents;
long ones (reference tables: agencies, products, cities...) only live here, so the values a
question names can still be found and written exactly in the SQL.

One SQLite file per database, <value_search.directory>/<db>.sqlite:
    value         one row per (table, column, value): normalized form and row count
    value_fts     FTS5 trigram index of the normalized forms (external content: value)
    value_column  one row per indexed column: number of values, inlined in the documents or not

A search runs in two steps (search_values):
    1. candidates — FTS5 matches ANY trigram of the term, ranked by bm25. Unlike a substring
                    match, it tolerates typos: 'casablnca' shares most trigrams with 'casablanca'.
    2. scoring    — rapidfuzz scores each candidate against the term; under value_search.min_score
                    it is dropped.
"""

import os
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from rapidfuzz import fuzz

from config import cfg

# FTS5 candidates scored by rapidfuzz, per searched term
MAX_CANDIDATES   = 200
# Longer values are not indexed: free text, not labels
MAX_VALUE_LENGTH = 200
# The trigram index needs at least 3 characters: shorter terms ('IT', 'RH') are matched exactly
TRIGRAM          = 3
# Closest values proposed for a value a query filters on but its column doesn't hold:
# a looser threshold than the search, any hint helps the correction
CLOSEST_VALUES    = 3
CLOSEST_MIN_SCORE = 60

SCHEMA = """
CREATE TABLE value (
    id          INTEGER PRIMARY KEY,
    table_name  TEXT    NOT NULL,
    column_name TEXT    NOT NULL,
    value       TEXT    NOT NULL,   -- as stored in the database
    norm        TEXT    NOT NULL,   -- normalize(value): what is searched
    row_count   INTEGER NOT NULL
);
CREATE INDEX value_by_column ON value (table_name, column_name, value);
CREATE INDEX value_by_norm   ON value (norm);

CREATE VIRTUAL TABLE value_fts USING fts5 (norm, content = 'value', content_rowid = 'id', tokenize = 'trigram');

CREATE TABLE value_column (
    table_name  TEXT    NOT NULL,
    column_name TEXT    NOT NULL,
    value_count INTEGER NOT NULL,
    inline      INTEGER NOT NULL,   -- 1 when the values are listed in the table document
    PRIMARY KEY (table_name, column_name)
);
"""


@dataclass(frozen=True)
class ValueMatch:
    table:  str
    column: str
    value:  str     # as stored in the database: the literal to write in the SQL
    score:  float   # rapidfuzz score against the searched term, 0-100
    rows:   int     # rows holding the value, breaks score ties


def value_index_path(db_name: str) -> Path:
    """<value_search.directory>/<db>.sqlite: the value index of a database, built by make index."""
    return Path(cfg.value_search.directory) / f"{db_name}.sqlite"


def normalize(text: str) -> str:
    """Searched form of a value or a term: no case, no accents, single spaces. 'São  Paulo' → 'sao paulo'."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped   = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.casefold().split())


# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

def write_value_index(db_name: str, values: dict[str, dict[str, dict[str, int]]],
                      inline: set[tuple[str, str]]):
    """
    Write the value index of a database, replacing the previous one.

    values: {table: {column: {value: row count}}}, the categorical values read from the database
    inline: the (table, column) whose values are listed in the table documents

    Written to a temporary file, then renamed: the running app never reads a half-written index.
    """
    path = value_index_path(db_name)
    tmp  = path.with_name(f"{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)

    rows = [
        (table, column, value, normalize(value), count)
        for table, columns in values.items()
        for column, counts in columns.items()
        for value, count in counts.items()
        if len(value) <= MAX_VALUE_LENGTH and normalize(value)
    ]
    indexed_columns = [
        (table, column, len(counts), (table, column) in inline)
        for table, columns in values.items()
        for column, counts in columns.items()
    ]

    with closing(sqlite3.connect(tmp)) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO value (table_name, column_name, value, norm, row_count) VALUES (?, ?, ?, ?, ?)", rows
        )
        conn.executemany("INSERT INTO value_column VALUES (?, ?, ?, ?)", indexed_columns)
        conn.execute("INSERT INTO value_fts (value_fts) VALUES ('rebuild')")
        conn.commit()

    os.replace(tmp, path)
    logger.info(f"[{db_name}] Indexed {len(rows)} values of {len(indexed_columns)} column(s) to {path.name}.")


# ------------------------------------------------------------
# Search
# ------------------------------------------------------------

@dataclass(frozen=True)
class MatchingValues:
    values:    dict[str, dict[str, list[str]]]   # {table: {column: [value, ...]}} to add to the schema, best first
    more:      dict[str, dict[str, int]]         # {table: {column: n}}: close values left out of `values`
    not_found: bool                              # terms searched, none close to any value


def find_matching_values(db_name: str, terms: list[str], tables: list[str]) -> MatchingValues:
    """
    Values of the given tables matching the terms of a question, in the columns that don't list
    their values in the table documents.

    Every value close to a term is kept, so planner can filter on them with = or IN: 'Santa' →
    'Santa Clara', 'Santa Cruz', 'Santa Monica'. Past value_search.top_k values per column, the
    others are only counted (more), and planner falls back on a LIKE.

    Listed columns are searched too, as a reference: for each term, only the matches within
    value_search.score_margin of its best one are kept, so a weaker match in another column is
    dropped. 'Mexico' → country 'Mexico' (100, listed already: nothing to add), not city
    'México D.F.' (90).

    not_found is only claimed when the tables have indexed columns: without any, nothing was searched.
    """
    columns = _indexed_columns(db_name, tables)   # {(table, column): listed in the documents}
    margin  = cfg.value_search.score_margin
    limit   = cfg.value_search.top_k

    values: dict[str, dict[str, list[str]]] = {}
    more:   dict[str, dict[str, int]]       = {}
    found   = False
    for term in terms if columns else []:
        # Every candidate above the threshold, not only the first top_k of each column
        matches = search_values(db_name, term, list(columns), top_k=MAX_CANDIDATES)
        if not matches:
            logger.debug(f"[{db_name}] No value close to '{term}'")
            continue
        found = True
        best  = matches[0].score

        close = {}   # {(table, column): [value, ...]} of this term
        for match in matches:
            if match.score >= best - margin and not columns[(match.table, match.column)]:
                close.setdefault((match.table, match.column), []).append(match.value)

        for (table, column), close_values in close.items():
            kept = values.setdefault(table, {}).setdefault(column, [])
            kept.extend(v for v in close_values if v not in kept)
            if len(kept) > limit:
                more.setdefault(table, {})[column] = more.get(table, {}).get(column, 0) + len(kept) - limit
                del kept[limit:]

    return MatchingValues(values, more, not_found=bool(columns and terms) and not found)


@dataclass(frozen=True)
class UnknownValue:
    table:   str
    column:  str
    value:   str         # filtered on by the query, absent from the column
    closest: list[str]   # values of the column close to it, best first


def find_unknown_values(db_name: str, filters: list[tuple[str, str, str]],
                        tables: list[str]) -> list[UnknownValue]:
    """
    Values a query filters on that their column doesn't hold — the query would silently
    return no row — each with the closest values the column holds.

    filters: (table, column, value), from lib/sql_tools.get_value_filters; table '' for an
             unqualified column, resolved among `tables` (the tables of the query) when only
             one of them has an indexed column of that name.
    Only the indexed columns are checked; a value too long to be indexed is never reported.
    """
    columns = _indexed_columns(db_name, [table.lower() for table in tables])
    if not columns:
        return []

    unknown = []
    with _connect(value_index_path(db_name)) as conn:
        for table, column, value in filters:
            if not table:
                owners = [t for t, c in columns if c == column]
                table  = owners[0] if len(owners) == 1 else ""
            if (table, column) not in columns or len(value) > MAX_VALUE_LENGTH:
                continue
            held = conn.execute(
                "SELECT 1 FROM value WHERE table_name = ? AND column_name = ? AND value = ? LIMIT 1",
                (table, column, value),
            ).fetchone()
            if not held:
                closest = search_values(db_name, value, [(table, column)], top_k=CLOSEST_VALUES,
                                        min_score=CLOSEST_MIN_SCORE)
                unknown.append(UnknownValue(table, column, value, [m.value for m in closest]))
    return unknown


def search_values(db_name: str, term: str, columns: list[tuple[str, str]] | None = None,
                  top_k: int | None = None, min_score: float | None = None) -> list[ValueMatch]:
    """
    Values close to a term of the question, best first, at most top_k per column.

    columns: the (table, column) to search, None for every indexed column.
    top_k and min_score default to value_search.top_k and value_search.min_score.
    Returns [] when the index of the database is not built (make index).
    """
    top_k     = top_k or cfg.value_search.top_k
    min_score = cfg.value_search.min_score if min_score is None else min_score
    norm      = normalize(term)
    path      = value_index_path(db_name)

    if not norm or columns == []:
        return []
    if not path.exists():
        logger.debug(f"[{db_name}] No value index at {path} — run make index.")
        return []

    with _connect(path) as conn:
        candidates = _candidates(conn, norm, columns)

    matches = sorted(
        (
            ValueMatch(table, column, value, fuzz.WRatio(norm, value_norm), rows)
            for table, column, value, value_norm, rows in candidates
        ),
        key=lambda m: (-m.score, -m.rows, m.value),
    )

    kept, per_column = [], Counter()
    for match in matches:
        if match.score >= min_score and per_column[(match.table, match.column)] < top_k:
            kept.append(match)
            per_column[(match.table, match.column)] += 1
    return kept


def _indexed_columns(db_name: str, tables: list[str]) -> dict[tuple[str, str], bool]:
    """
    (table, column) of the given tables in the value index, mapped to True when their values
    are listed in the table documents. {} when the index is not built.
    """
    path = value_index_path(db_name)
    if not tables or not path.exists():
        return {}
    with _connect(path) as conn:
        rows = conn.execute(
            f"SELECT table_name, column_name, inline FROM value_column "
            f"WHERE table_name IN ({', '.join('?' * len(tables))})",
            tables,
        ).fetchall()
    return {(table, column): bool(inline) for table, column, inline in rows}


def _connect(path: Path) -> closing[sqlite3.Connection]:
    """Read-only connection, closed on exit: one per call, so threads never share one."""
    return closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True))


def _candidates(conn: sqlite3.Connection, norm: str,
                columns: list[tuple[str, str]] | None) -> list[tuple[str, str, str, str, int]]:
    """
    (table, column, value, norm, row count) of the values worth scoring, in the given columns:
    the MAX_CANDIDATES best bm25 matches of any trigram of the term, or the exact matches of a short term.
    """
    params: list = []
    column_filter = ""
    if columns is not None:
        column_filter = f"AND (v.table_name, v.column_name) IN (VALUES {', '.join(['(?, ?)'] * len(columns))})"
        params        = [name for column in columns for name in column]

    if len(norm) < TRIGRAM:
        return conn.execute(
            f"SELECT v.table_name, v.column_name, v.value, v.norm, v.row_count FROM value AS v "
            f"WHERE v.norm = ? {column_filter}",
            [norm, *params],
        ).fetchall()

    # Any trigram of the term, each one quoted as an FTS5 string ("" escapes a double quote)
    trigrams = dict.fromkeys(norm[i:i + TRIGRAM] for i in range(len(norm) - TRIGRAM + 1))
    query    = " OR ".join('"' + trigram.replace('"', '""') + '"' for trigram in trigrams)

    return conn.execute(
        f"SELECT v.table_name, v.column_name, v.value, v.norm, v.row_count "
        f"FROM value_fts JOIN value AS v ON v.id = value_fts.rowid "
        f"WHERE value_fts MATCH ? {column_filter} "
        f"ORDER BY bm25(value_fts) LIMIT ?",
        [query, *params, MAX_CANDIDATES],
    ).fetchall()
