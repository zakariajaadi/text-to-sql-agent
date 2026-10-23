# offline/studio/app.py

"""
Studio of the semantic layer: a web page to run the pipeline steps and see what they write, instead
of patch scripts. Step 1, profiling: the tables and their status at each step, profile the ones chosen
or missing, read the profile of a table (declared type against what the data holds), make an exception
to what it detects, hide the personal data of the whole database.

    PYTHONPATH=src python -m uvicorn offline.studio.app:app --port 8100     (see `make studio`)

The YAML files of semantic_layer/<db>/ stay the source of truth: the CLI and the Makefile keep working,
the studio reads the files at each request and writes tables.yaml only on a click of the page.
It shows shapes and counts of the data, never a value. The database is only read to list the tables.
"""

import threading
import time
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import inspect as sa_inspect

from config import SEMANTIC_LAYER_DIR, cfg
from lib.database import build_database
from lib.semantics import load_fk_relations, load_profile, load_semantics, tables_path
from offline.semantic_gen.utils import dump_table_semantics
from offline.studio import state
from offline.studio.jobs import JobRunner, profile_command

STATIC = Path(__file__).parent / "static"

app    = FastAPI(title="Semantic layer studio")
runner = JobRunner()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ------------------------------------------------------------
# Tables of a database (read once, then on demand: a metastore call per table)
# ------------------------------------------------------------

_databases: dict = {}
_listings:  dict[str, dict] = {}
_lock = threading.Lock()


def _db_config(db: str):
    if db not in {d.name for d in cfg.databases}:
        raise HTTPException(404, f"Unknown database: {db}")
    return cfg.get_db(db)


def _listing(db: str, refresh: bool = False) -> dict:
    """
    The usable tables of the database and their number of columns. When the database can't be
    reached, the tables known from the files (profile.yaml, tables.yaml), with the error to show.
    """
    with _lock:
        if not refresh and db in _listings:
            return _listings[db]
        try:
            if db not in _databases:
                _databases[db] = build_database(_db_config(db))
            database  = _databases[db]
            inspector = sa_inspect(database._engine)
            tables    = database.get_usable_table_names()
            listing   = {"tables": tables, "source": "database", "error": None, "columns": {
                t: [{"name": c["name"], "type": _type_name(c["type"])} for c in inspector.get_columns(t)] for t in tables
            }}
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[studio] {db}: tables read from the files, the database is unreachable — {e}")
            _databases.pop(db, None)
            known   = list(dict.fromkeys([*load_profile(db), *load_semantics(db)]))
            listing = {"tables": known, "columns": {}, "source": "files", "error": str(e).splitlines()[0][:300]}
        listing["read_at"] = time.time()
        _listings[db] = listing
        return listing


def _type_name(column_type) -> str:
    """The SQL type of a column (VARCHAR, DECIMAL(10, 2)...); its class name when a dialect type can't print itself."""
    try:
        return str(column_type)
    except Exception:
        return type(column_type).__name__


def _version(db: str) -> str:
    """
    Version of tables.yaml (its modification time): a review saved over a newer file is refused.
    A string: nanoseconds go past the integers a JavaScript number holds exactly.
    """
    path = tables_path(db)
    return str(path.stat().st_mtime_ns if path.exists() else 0)


def _load_metrics(db: str) -> dict:
    path = SEMANTIC_LAYER_DIR / db / "metrics.yaml"
    return (yaml.safe_load(path.read_text()) if path.exists() else None) or {}


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/databases")
def databases():
    return [{"name": d.name, "dialect": d.dialect, "personal": d.exclude_personal_data} for d in cfg.databases]


@app.get("/api/db/{db}/tables")
def tables(db: str, refresh: bool = False):
    """The home screen: every table with its status at each step, and the counts of each step."""
    db_cfg    = _db_config(db)
    listing   = _listing(db, refresh)
    profile   = load_profile(db)
    semantics = load_semantics(db)
    relations = load_fk_relations(db)
    metrics   = _load_metrics(db)
    counts = {table: len(columns) for table, columns in listing["columns"].items()}
    rows = state.table_rows(listing["tables"], counts, profile, semantics, relations, metrics,
                            db_cfg.exclude_personal_data)
    count = lambda step, status: sum(r["status"][step] == status for r in rows)  # noqa: E731
    personal = [s for columns in state.personal_columns(profile, semantics).values() for s in columns.values()]
    return {
        "db": db, "source": listing["source"], "error": listing["error"], "read_at": listing["read_at"],
        "version": _version(db), "tables": rows,
        "personal": {visibility: personal.count(visibility) for visibility in ("hidden", "kept", "visible")},
        "summary": {
            "tables":    len(rows),
            "profiled":  len(rows) - count("profile", state.NONE),
            "described": count("semantics", state.DONE),
            "relations": len(relations),
            "metrics":   len([name for name in metrics if name != "ignore"]),
        },
    }


@app.get("/api/db/{db}/tables/{table}")
def table_detail(db: str, table: str):
    """The profile of a table and its proposals, for the detail panel."""
    db_cfg  = _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Unknown table: {table}")
    db_columns = listing["columns"].get(table)
    detail = state.table_detail(table, load_profile(db).get(table), load_semantics(db).get(table),
                                db_cfg.exclude_personal_data, db_columns)
    return {**detail, "columns": len(db_columns) if db_columns else None, "version": _version(db)}


class ProfileRequest(BaseModel):
    tables: list[str] | None = None
    missing: bool = False


@app.post("/api/db/{db}/profile")
def profile(db: str, request: ProfileRequest):
    """Profile the tables chosen, or the ones missing from the profile (make profile ARGS=...)."""
    _db_config(db)
    if not request.tables and not request.missing:
        raise HTTPException(400, "Choose tables, or the missing ones")
    if unknown := sorted(set(request.tables or []) - set(_listing(db)["tables"])):
        raise HTTPException(400, f"Unknown table(s): {', '.join(unknown)}")
    try:
        job = runner.start(db, "profile", profile_command(db, request.tables, request.missing), request.tables)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return job.to_dict()


def _update_semantics(db: str, version: str, change) -> None:
    """
    Rewrite tables.yaml with change(semantics), unless it changed since the page read it (`version`).
    Written at once (a temporary file renamed): a run of the CLI may read it at the same time.
    """
    with _lock:
        if version != _version(db):
            raise HTTPException(409, "tables.yaml changed since the page read it: reload the table")
        semantics = change(load_semantics(db))
        path = tables_path(db)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(dump_table_semantics(semantics))
        temporary.replace(path)


def _profiled(db: str, table: str) -> dict:
    _db_config(db)
    if not (facts := load_profile(db).get(table)):
        raise HTTPException(404, f"{table} is not profiled")
    return facts


class ColumnChange(BaseModel):
    column: str
    action: Literal["format", "text", "hide", "show"]
    version: str


@app.post("/api/db/{db}/tables/{table}/columns")
def column_change(db: str, table: str, request: ColumnChange):
    """
    An exception to what the profile detects in a column, from the pill of the page: read it in the
    detected format or keep it as text, hide it or keep it visible on purpose.
    """
    facts  = _profiled(db, table)
    column = request.column
    if request.action in ("format", "text"):
        if not (value := (facts.get("formats") or {}).get(column)):
            raise HTTPException(400, f"The profile detects no format in {table}.{column}")
        change = lambda entry: state.set_format(entry, column, value if request.action == "format" else None)  # noqa: E731
    else:
        if column not in (facts.get("exclude") or {}):
            raise HTTPException(400, f"The profile proposes no exclusion of {table}.{column}")
        change = lambda entry: state.set_excluded(entry, [column], request.action == "hide")  # noqa: E731
    _update_semantics(db, request.version, lambda semantics: {**semantics, table: change(semantics.get(table))})
    logger.info(f"[studio] {db}.{table}.{column}: {request.action}")
    return table_detail(db, table)


class PersonalData(BaseModel):
    hide: bool
    version: str


@app.post("/api/db/{db}/personal")
def personal_data(db: str, request: PersonalData):
    """Hide the personal data of every profiled table (the exceptions kept visible), or show it again."""
    _db_config(db)
    profile = load_profile(db)
    _update_semantics(db, request.version, lambda semantics: state.hide_personal(profile, semantics, request.hide))
    logger.info(f"[studio] {db}: personal data {'hidden' if request.hide else 'shown'} in every table")
    return tables(db)


@app.get("/api/job")
def job(since: int = 0):
    """The last job (running or ended), with its output lines from number `since` on."""
    return runner.job.to_dict(since) if runner.job else None


@app.post("/api/job/stop")
def stop():
    runner.stop()
    return job()
