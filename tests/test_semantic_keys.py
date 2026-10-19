"""exclude_columns and data_as_of of tables.yaml, in the rendered document (src/lib/semantics.py)."""

from config import DatabaseConfig
from lib.column_pruning import get_column_names
from lib.semantics import excluded_columns, manual_semantics, render_table

DDL = """CREATE TABLE `client` (
\t`clientid` INT,
\t`name` STRING,
\t`segment` STRING,
\t`validated_on` STRING
)

/*
3 rows from client table:
clientid\tname\tsegment\tvalidated_on
1\tJohn Doe\tRetail\t
2\tJane Roe\tCorporate\t
*/"""

SEMANTICS = {
    "description":     "One row per client.",
    "data_as_of":      "2026-06-16",
    "exclude_columns": ["name", "validated_on"],
    "columns":         {"clientid": "Client number.", "name": "Client name.", "segment": "Client segment."},
}


def test_excluded_columns_are_out_of_the_body_and_the_sample_rows():
    doc = render_table("client", DDL, SEMANTICS, [])

    assert get_column_names(doc) == ["clientid", "segment"]
    assert "John Doe" not in doc and "name" not in doc.split("*/")[0].split("/*")[1]
    assert "1\tRetail" in doc
    assert "`segment` STRING,  -- Client segment." in doc


def test_data_as_of_is_a_header_line():
    lines = render_table("client", DDL, SEMANTICS, []).splitlines()
    assert lines[:3] == ["-- Table: client", "-- Description: One row per client.", "-- Data as of: 2026-06-16"]


def test_without_the_keys_the_document_is_unchanged():
    doc = render_table("client", DDL, {"description": "One row per client."}, [])
    assert get_column_names(doc) == ["clientid", "name", "segment", "validated_on"]
    assert "Data as of" not in doc


def test_excluded_columns_of_a_missing_table():
    assert excluded_columns(None) == set()
    assert excluded_columns(SEMANTICS) == {"name", "validated_on"}


def test_sample_rows_default_and_override():
    assert DatabaseConfig(name="Chinook", dialect="hive").sample_rows == 3
    assert DatabaseConfig(name="Sales", dialect="impala", sample_rows=0).sample_rows == 0


def test_semantic_language_defaults_to_english():
    assert DatabaseConfig(name="Chinook", dialect="hive").semantic_language == "English"
    assert DatabaseConfig(name="Sales", dialect="impala", semantic_language="French").semantic_language == "French"


def test_manual_semantics_drops_only_the_generated_keys():
    entry = dict(SEMANTICS, categorical_columns=["segment"], formats={"validated_on": "date dd/MM/yyyy"})
    assert manual_semantics(entry) == {
        "data_as_of":      "2026-06-16",
        "exclude_columns": ["name", "validated_on"],
        "formats":         {"validated_on": "date dd/MM/yyyy"},
    }


def test_the_generator_prompts_take_the_language():
    from offline.semantic_gen.prompts import METRICS_PROMPT, TABLE_SEMANTICS_PROMPT
    assert "{language}" in TABLE_SEMANTICS_PROMPT and "{language}" in METRICS_PROMPT
