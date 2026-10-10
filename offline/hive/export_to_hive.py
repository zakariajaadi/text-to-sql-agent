"""
Export a SQLite database to TSV files + a Hive load script (load.hql).

Usage:
    uv run python scripts/export_to_hive.py Chinook
"""

import sqlite3
import sys
from pathlib import Path

TYPE_MAP = {"INTEGER": "INT", "REAL": "DOUBLE", "DATETIME": "TIMESTAMP", "DATE": "DATE"}


def hive_type(sqlite_type: str) -> str:
    t = sqlite_type.upper()
    if t.startswith("NUMERIC"):
        return t.replace("NUMERIC", "DECIMAL") if "(" in t else "DECIMAL(10,2)"
    return TYPE_MAP.get(t, "STRING")


def clean(value) -> str:
    # \N is Hive's default NULL marker in text files
    if value is None:
        return r"\N"
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


db_name = sys.argv[1]
hive_db = db_name.lower()
out_dir = Path("data/hive_load") / hive_db
out_dir.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(f"data/{db_name}.db")
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
)]

hql = [f"CREATE DATABASE IF NOT EXISTS {hive_db};", f"USE {hive_db};"]

for table in tables:
    # BLOB columns are useless for text-to-SQL
    cols  = [c for c in conn.execute(f'PRAGMA table_info("{table}")') if c[2].upper() != "BLOB"]
    name  = table.lower().replace(" ", "_")
    names = ", ".join(f'"{c[1]}"' for c in cols)

    with open(out_dir / f"{name}.tsv", "w") as f:
        for row in conn.execute(f'SELECT {names} FROM "{table}"'):
            f.write("\t".join(clean(v) for v in row) + "\n")

    col_defs = ",\n  ".join(f"`{c[1]}` {hive_type(c[2])}" for c in cols)
    hql += [
        f"DROP TABLE IF EXISTS {name};",
        f"CREATE TABLE {name} (\n  {col_defs}\n)\n"
        f"ROW FORMAT DELIMITED FIELDS TERMINATED BY '\\t' STORED AS TEXTFILE;",
        f"LOAD DATA LOCAL INPATH '/tmp/hive_load/{hive_db}/{name}.tsv' OVERWRITE INTO TABLE {name};",
    ]

(out_dir / "load.hql").write_text("\n".join(hql) + "\n")
print(f"{len(tables)} tables exported to {out_dir}")