# offline/semantic_gen/profiling.py

"""
Data profile of a database: semantic_layer/<db>/profile.yaml, measured on the data, no LLM.
Read by the generators so the semantic layer states facts instead of guesses:

    - row facts: rows of each table and distinct values of its identifier-like columns.
      Fed to the semantics prompt ("several rows per account"), and the source of the
      relations found without an LLM (a key unique in one table, present in others).
    - text columns: shapes of their values (digits → 9, other letters → a) on a sample,
      giving the `formats` of dates and numbers stored as text, and the empty columns.
    - personal data: columns whose name says so (names, ids, contacts, free text, authors),
      or whose values look like logins or emails.

Formats and exclusions are proposals: the semantics generator writes them into tables.yaml
only for a table that has none yet, for review. Each run replaces the entries of the tables it
measures (every table, or the ones asked for) and keeps the others.

Pipeline order: profile → semantics → relations → metrics → (review) → index   (see offline/cli.py)
"""

import re

import yaml
from langchain_community.utilities import SQLDatabase
from loguru import logger
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from lib.semantics import load_profile, profile_path

PROFILE_HEADER = """\
# Data profile, measured by `make profile` (no LLM): the tables measured replace their entry, never edited.
# Read by `make semantics` (row facts, proposed formats and exclusions) and `make relations`.
#   rows, keys          : row count, and per identifier-like column its non-null and distinct values
#   text_columns        : filled share and main value shapes (digits → 9, letters → a) on a sample
#   formats, exclude    : what the profile proposes for tables.yaml, with the reason of each exclusion

"""

SAMPLE_ROWS       = 100_000   # rows read to compute the shapes of the text columns
MAX_KEYS          = 8         # identifier-like columns counted per table
MAX_SHAPES        = 6         # shapes kept per column in the profile
EMPTY_FILL        = 0.01      # a column filled below this share is proposed for exclusion
FORMAT_SHARE      = 0.9       # a format is proposed when it covers this share of the filled values

# Identifier-like column names: numero_x, id_x, x_id, xid, x_host...
KEY_NAME  = re.compile(r"^(numero|num|nr|id|n)_|(id|_host)$")
# Codes and identifiers stored as digits: never proposed as numbers
CODE_NAME = re.compile(r"(^|_)(code|cod|cd|id|numero|num|nr|n|ref|matricule|rowd|row)(_|$)|id$")
# Personal data, read from the column name
PERSONAL_NAME = re.compile(
    r"(^|_)(nom|noms|prenom|prenoms|raison_?sociale?|denomination|cin|passeport|email|mail\d?|gsm|tel|"
    r"telephone|fax|adresse|adr|contact|commentaires?|login|porteur|iban|rib|swift|masque_carte|"
    r"num_?crt|signataire|beneficiaire|intitule|intit\d|employeur|lieu_naissance|"
    r"(cree|creee|crea|modifie|modifiee|modif|maj|valide|validee|saisi|saisie)_?par|"
    r"first_?name|last_?name|phone|address|birth_?date|ssn|(created|modified|updated)_?by)(_|$)"
    # Birth date, misspelled too (date_naisance)
    r"|(^|_)(date|dt)_?nais"
    # A name glued to the next word (nomintervenants, prenomclient), not nombre, nominal, nomenclature
    r"|(^|_)(pre)?noms?(?!bre|ina|encl|ade)[a-z]"
)
# Values shaped like a login (first.last, first-name.last) or an email: personal data whatever the name
PERSONAL_SHAPE = re.compile(r"^a([ -]a)*\.a([ -]a)*9*$|^[a9.+-]+@[a9.-]+\.a$")
PERSONAL_SHARE = 0.5   # share of the filled values shaped like one, a user column mixing logins and codes

# Shapes (digits → 9) of dates stored as text → their Java pattern
DATE_SHAPES = [
    (re.compile(r"^99/99/9999$"),                         "dd/MM/yyyy"),
    (re.compile(r"^99/99/9999 99:99:99(\.9+)?$"),         "dd/MM/yyyy HH:mm:ss"),
    (re.compile(r"^99-99-9999$"),                         "dd-MM-yyyy"),
    (re.compile(r"^9999-99-99$"),                         "yyyy-MM-dd"),
    (re.compile(r"^9999-99-99 99:99:99(\.9+)?$"),         "yyyy-MM-dd HH:mm:ss"),
    (re.compile(r"^9999/99/99$"),                         "yyyy/MM/dd"),
    # DB2 timestamps, written with their microseconds or without
    (re.compile(r"^9999-99-99-99\.99\.99$"),              "yyyy-MM-dd-HH.mm.ss"),
    (re.compile(r"^9999-99-99-99\.99\.99\.999999$"),      "yyyy-MM-dd-HH.mm.ss.SSSSSS"),
]
# Numbers stored as text: 12, 12.5, 12,5, .5, -3, 1,5E+5, 1.5E5 (the exponent letter is an a).
# An exponent follows digits and has a sign, or a decimal mantissa: A123 or 12A345 is a code
NUMBER_SHAPE = re.compile(r"^[+-]?(9+([.,]9+)?|[.,]9+)(a[+-]9+)?$|^[+-]?9*[.,]9+a9+$")
WHOLE_NUMBER = re.compile(r"^[+-]?9+$")


# ------------------------------------------------------------
# Profile
# ------------------------------------------------------------

def profile_database(db: SQLDatabase, db_name: str, tables: list[str] | None = None,
                     missing: bool = False) -> dict:
    """
    Profile the usable tables of the database, one table after the other (a Hive or Impala job
    at a time), and merge them into semantic_layer/<db>/profile.yaml: every table, only `tables`,
    and with `missing` only the ones the profile lacks (see tables_to_profile). The entries of the
    other tables are kept, those of tables no longer usable dropped. A table whose queries fail
    keeps its previous entry, if any, with a warning: the profile is an enrichment, never required.
    """
    usable   = db.get_usable_table_names()
    existing = load_profile(db_name)
    todo     = tables_to_profile(usable, existing, tables, missing)
    if not todo:
        logger.info(f"[{db_name}] No table to profile: every one asked for is in {profile_path(db_name).name}")
        return existing

    inspector = sa_inspect(db._engine)
    quote     = db._engine.dialect.identifier_preparer.quote
    measured  = {}
    for table in todo:
        columns = inspector.get_columns(table)
        logger.info(f"[{db_name}] Profiling: {table}")
        try:
            measured[table] = profile_table(db, table, columns, quote)
        except Exception as e:
            logger.warning(f"[{db_name}] Not profiled: {table} — {str(e).splitlines()[0][:200]}")

    profile = merge_profiles(usable, existing, measured)
    if dropped := sorted(set(existing) - set(profile)):
        logger.info(f"[{db_name}] Dropped from the profile, no longer usable: {', '.join(dropped)}")
    path = profile_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROFILE_HEADER + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True, width=1000))
    logger.info(f"[{db_name}] {len(measured)} table(s) profiled, {len(profile)} in the profile → {path}")
    return profile


def tables_to_profile(usable: list[str], existing: dict, tables: list[str] | None, missing: bool) -> list[str]:
    """
    The tables to measure, in the order of the database: `tables` (all usable ones when None),
    and with `missing` only those absent from the existing profile. Raises ValueError on a
    table that is not usable (a typo would silently measure nothing).
    """
    if tables and (unknown := sorted(set(tables) - set(usable))):
        raise ValueError(f"Unknown or excluded table(s): {', '.join(unknown)}")
    return [t for t in usable if (not tables or t in tables) and not (missing and t in existing)]


def merge_profiles(usable: list[str], existing: dict, measured: dict) -> dict:
    """The measured entries over the existing ones, for the usable tables, in the order of the database."""
    return {t: measured.get(t, existing.get(t)) for t in usable if t in measured or t in existing}


def profile_table(db: SQLDatabase, table: str, columns: list[dict], quote) -> dict:
    """Row facts, text column shapes and the proposals of one table."""
    names = [c["name"] for c in columns]
    texts = [c["name"] for c in columns if _is_text(c["type"])]
    keys  = [n for n in names if KEY_NAME.search(n.lower())][:MAX_KEYS]

    rows, key_counts = _count_rows(db, table, keys, quote)
    shapes = {column: _text_shapes(db, table, column, quote) for column in texts}

    entry = {"rows": rows, "keys": key_counts, "text_columns": {}}
    formats, exclude = {}, {}
    for column, column_shapes in shapes.items():
        facts = describe_shapes(column_shapes)
        entry["text_columns"][column] = facts
        if facts["filled"] < EMPTY_FILL:
            exclude[column] = "empty"
        elif personal_values(column_shapes):
            exclude[column] = "personal data"
        elif fmt := text_format(column_shapes, column):
            formats[column] = fmt
    for column in names:
        if column not in exclude and PERSONAL_NAME.search(column.lower()):
            exclude[column] = "personal data"
    if formats:
        entry["formats"] = formats
    if exclude:
        entry["exclude"] = exclude
    return entry


# ------------------------------------------------------------
# Queries
# ------------------------------------------------------------

def _count_rows(db: SQLDatabase, table: str, keys: list[str], quote) -> tuple[int, dict]:
    """Row count, and the non-null and distinct values of each key column (one full scan)."""
    parts = ["COUNT(*)"]
    for key in keys:
        parts += [f"COUNT({quote(key)})", f"COUNT(DISTINCT {quote(key)})"]
    with db._engine.connect() as conn:
        values = conn.execute(text(f"SELECT {', '.join(parts)} FROM {quote(table)}")).fetchone()
    return int(values[0]), {
        key: {"non_null": int(values[1 + 2 * i]), "distinct": int(values[2 + 2 * i])}
        for i, key in enumerate(keys)
    }


def key_overlap(db: SQLDatabase, source: str, target: str, sample: int = 10_000) -> float | None:
    """
    Share of the distinct values of source (table.column), on a sample, found in target: the same
    column name in two tables is a relation only when the values match (numero_contrat of a card
    is not the one of a loan). None when source holds no value: nothing to check.
    """
    quote = db._engine.dialect.identifier_preparer.quote
    with db._engine.connect() as conn:
        total, found = conn.execute(text(key_overlap_sql(source, target, quote, sample))).fetchone()
    return found / total if total else None


def key_overlap_sql(source: str, target: str, quote, sample: int) -> str:
    """The query of key_overlap: distinct sampled source values, and how many of them target holds."""
    (s_table, s_col), (t_table, t_col) = source.split("."), target.split(".")
    s_key, t_key = _key_sql(quote(s_col)), _key_sql(quote(t_col))
    return (
        f"SELECT COUNT(*), COUNT(t.k) FROM "
        f"(SELECT DISTINCT {s_key} AS k FROM {quote(s_table)} "
        f"WHERE {quote(s_col)} IS NOT NULL LIMIT {sample}) s "
        f"LEFT JOIN (SELECT DISTINCT {t_key} AS k FROM {quote(t_table)}) t ON s.k = t.k"
    )


def _key_sql(column: str) -> str:
    """
    A key value as comparable text: the two columns may be stored as int and string, and a string
    of digits may be padded with zeros or blanks ('01234567' is 1234567). Digits only (up to 18,
    a BIGINT) → the number, written without its leading zeros; anything else → the trimmed text.
    """
    value = f"TRIM(CAST({column} AS STRING))"
    return f"CASE WHEN {value} RLIKE '^[0-9]{{1,18}}$' THEN CAST(CAST({value} AS BIGINT) AS STRING) ELSE {value} END"


def _shape_sql(column: str) -> str:
    """SQL of the shape of a value: digits → 9, every run of other letters → a (separators kept)."""
    return f"REGEXP_REPLACE(REGEXP_REPLACE({column}, '[0-9]', '9'), '[^9/.:,+@ -]+', 'a')"


def _text_shapes(db: SQLDatabase, table: str, column: str, quote) -> dict:
    """
    {shape: count} of a text column on the first SAMPLE_ROWS rows of the table (None: NULL).
    One light query per column: a UNION of many columns runs out of memory on a small Hive.
    """
    shape = _shape_sql(quote(column))
    query = (f"SELECT {shape} AS shape, COUNT(*) AS n "
             f"FROM (SELECT {quote(column)} FROM {quote(table)} LIMIT {SAMPLE_ROWS}) s GROUP BY {shape}")
    with db._engine.connect() as conn:
        return {value: int(n) for value, n in conn.execute(text(query))}


# ------------------------------------------------------------
# Reading the shapes (pure functions, unit-tested)
# ------------------------------------------------------------

def _is_text(column_type) -> bool:
    """Text column, by the name of its type (Hive types don't subclass SQLAlchemy's)."""
    name = type(column_type).__name__.lower()
    return any(t in name for t in ("string", "char", "text"))


def _is_empty(shape: str | None) -> bool:
    """NULL, empty or only spaces (a fixed-width column padded with blanks)."""
    return shape is None or not shape.strip()


def describe_shapes(shapes: dict) -> dict:
    """Filled share (neither NULL, empty nor blank) and the main shapes of a column, as shares."""
    total  = sum(shapes.values()) or 1
    filled = sum(n for shape, n in shapes.items() if not _is_empty(shape))
    main   = sorted(((s, n) for s, n in shapes.items() if not _is_empty(s)), key=lambda x: -x[1])
    return {
        "filled": round(filled / total, 3),
        "shapes": {shape: round(n / total, 3) for shape, n in main[:MAX_SHAPES]},
    }


def text_format(shapes: dict, column: str) -> str | None:
    """
    The `formats` value of a text column: "date <pattern>" or "decimal" when one of them covers
    FORMAT_SHARE of the filled values, None otherwise. Codes and identifiers stay text (leading
    zeros, never summed): a column named like one, or whole numbers that always have the same
    number of digits (a quantity varies in length: 7, 30, 120).
    """
    filled = {s: n for s, n in shapes.items() if not _is_empty(s)}
    total  = sum(filled.values())
    if not total:
        return None

    kinds = {}
    for shape, n in filled.items():
        kind = next((f"date {pattern}" for regex, pattern in DATE_SHAPES if regex.match(shape)), None)
        if kind is None and NUMBER_SHAPE.match(shape):
            kind = "decimal"
        if kind:
            kinds[kind] = kinds.get(kind, 0) + n

    if not kinds:
        return None
    kind, n = max(kinds.items(), key=lambda item: item[1])
    if n / total < FORMAT_SHARE:
        return None
    if kind == "decimal":
        if CODE_NAME.search(column.lower()):
            return None
        numbers = [s for s in filled if NUMBER_SHAPE.match(s)]
        if all(WHOLE_NUMBER.match(s) for s in numbers) and len({len(s.lstrip("+-")) for s in numbers}) == 1:
            return None
    return kind


def personal_values(shapes: dict) -> bool:
    """Values shaped like logins or emails (PERSONAL_SHAPE) for PERSONAL_SHARE of the filled ones."""
    filled = {s: n for s, n in shapes.items() if not _is_empty(s)}
    total  = sum(filled.values())
    return bool(total) and sum(n for s, n in filled.items() if PERSONAL_SHAPE.match(s)) / total >= PERSONAL_SHARE


def row_facts(table_profile: dict | None) -> str:
    """The row facts of a table, as lines for the semantics prompt."""
    if not table_profile:
        return "No row facts measured (run `make profile`)."
    rows  = table_profile["rows"]
    lines = [f"- {rows} rows"]
    for key, counts in (table_profile.get("keys") or {}).items():
        distinct = counts["distinct"]
        if distinct and counts["non_null"] == distinct and distinct >= rows * 0.99:
            lines.append(f"- {key}: {distinct} distinct values, one row per {key}")
        elif distinct:
            lines.append(f"- {key}: {distinct} distinct values, {counts['non_null'] / distinct:.1f} rows per {key} on average")
    return "\n".join(lines)


def key_relations(profile: dict) -> list[tuple[str, str]]:
    """
    Relations found without an LLM: a key column unique in a table (its hub, the largest one when
    several) and present under the same name in other tables → (table.key, hub.key).
    """
    hubs = {}
    for table, entry in profile.items():
        rows = entry.get("rows") or 0
        for key, counts in (entry.get("keys") or {}).items():
            unique = counts["distinct"] and counts["non_null"] == counts["distinct"] >= rows * 0.99
            if unique and counts["distinct"] > hubs.get(key, ("", 0))[1]:
                hubs[key] = (table, counts["distinct"])

    relations = []
    for table, entry in profile.items():
        for key in entry.get("keys") or {}:
            hub = hubs.get(key, (None,))[0]
            if hub and hub != table:
                relations.append((f"{table}.{key}", f"{hub}.{key}"))
    return relations
