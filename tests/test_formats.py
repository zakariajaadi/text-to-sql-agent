"""Columns stored as text (`formats` in tables.yaml): the note rendered in their comment (src/lib/semantics.py)."""

import pytest

from lib.semantics import render_table

DDL = ("CREATE TABLE `client` (\n"
       "\t`clientid` INT,\n"
       "\t`entry_date` STRING,\n"
       "\t`closed_at` STRING,\n"
       "\t`deposit` STRING,\n"
       "\t`segment` STRING\n"
       ")")


def column_line(doc: str, column: str) -> str:
    return next(line for line in doc.splitlines() if f"`{column}`" in line)


def test_format_notes_are_appended_to_the_comment():
    doc = render_table("client", DDL, {
        "description": "One row per client.",
        "columns": {"entry_date": "Date the client joined.", "segment": "Client segment."},
        "formats": {
            "entry_date": "date dd/MM/yyyy",
            "closed_at":  "date yyyy-MM-dd HH:mm:ss",   # no description: the note alone
            "deposit":    "decimal",
        },
    }, [], categorical_values={"segment": ["Retail", "Corporate"]})

    assert column_line(doc, "entry_date").endswith("-- Date the client joined. [stored as text: date dd/MM/yyyy]")
    assert column_line(doc, "closed_at").endswith("-- [stored as text: date yyyy-MM-dd HH:mm:ss]")
    assert column_line(doc, "deposit").endswith("-- [stored as text: decimal number]")
    # Columns without a format are left as they were
    assert column_line(doc, "segment").endswith("-- Client segment. Values: 'Retail', 'Corporate'")
    assert column_line(doc, "clientid").strip() == "`clientid` INT,"


@pytest.mark.parametrize("column_format", ["date", "datetime dd/MM/yyyy", "decimal 2", "number"])
def test_an_unknown_format_is_rejected(column_format):
    with pytest.raises(ValueError, match="Unknown format"):
        render_table("client", DDL, {"description": "x", "formats": {"entry_date": column_format}}, [])
