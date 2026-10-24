# offline/studio/app.py

"""
Studio of the semantic layer: a web page to run the pipeline steps and see what they write, instead
of patch scripts (Profiling, Semantics, Relations, Metrics, then Index: what the agent reads, and its eval). Step 1, profiling: the tables and their status at each step, profile the ones chosen
or missing, read the profile of a table (declared type against what the data holds), make an exception
to what it detects, hide the personal data of the whole database.

    PYTHONPATH=src python -m uvicorn offline.studio.app:app --port 8100     (see `make studio`)

The YAML files of semantic_layer/<db>/ stay the source of truth: the CLI and the Makefile keep working,
the studio reads the files at each request and writes tables.yaml only on a click of the page.
It shows shapes and counts of the data, never a value. The database is read to list the tables, to
measure a relation added and to test a metric (counts only).
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from config import SEMANTIC_LAYER_DIR, cfg
from lib.database import build_database
from lib.metrics import metrics_path, read_metrics_file
from lib.semantics import excluded_columns, load_fk_relations, load_profile, load_semantics, tables_path
from lib.value_index import value_index_path
from offline.eval.dashboard import DATASETS_DIR, RESULTS_DIR, build_dashboard
from offline.indexer.trace import read_trace
from offline.semantic_gen.profiling import key_overlap, key_spread
from offline.semantic_gen.utils import (
    dump_table_semantics,
    load_relations_file,
    relations_path,
    write_metrics_file,
    write_relations_file,
)
from offline.studio import state
from offline.studio.index import document_view, eval_run, index_view
from offline.studio.jobs import (
    JobRunner,
    eval_command,
    index_command,
    metrics_command,
    profile_command,
    relations_command,
    semantics_command,
)
from offline.studio.metrics import (
    edit_metric,
    ignore_metric,
    last_run,
    metric_row,
    metrics_view,
    measure_key,
    measure_queries,
    read_measures,
)
from offline.studio.relations import (
    add_relation,
    fk_graph,
    ignore_relation,
    join_path,
    relations_view,
)

STATIC = Path(__file__).parent / "static"

app    = FastAPI(title="Semantic layer studio")
runner = JobRunner()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def revalidate(request, call_next):
    """The page and its files are checked again at each load: after an update, never an old script with a new page."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


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
    names = {table: [c["name"] for c in columns] for table, columns in listing["columns"].items()}
    rows = state.table_rows(listing["tables"], names, profile, semantics, relations, metrics,
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
            "described": len(rows) - count("semantics", state.NONE),
            "partial":   count("semantics", state.PARTIAL),
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


@app.get("/api/db/{db}/tables/{table}/semantics")
def table_semantics(db: str, table: str):
    """A table as the Semantics screen shows it: descriptions, data_as_of, categorical columns."""
    _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Unknown table: {table}")
    detail = state.semantics_detail(table, load_semantics(db).get(table), load_profile(db).get(table),
                                    listing["columns"].get(table))
    return {**detail, "version": _version(db)}


class SemanticsEdit(BaseModel):
    field: Literal["description", "data_as_of", "column", "categorical", "read"]
    value: str | bool | None = None
    column: str | None = None
    version: str


@app.post("/api/db/{db}/tables/{table}/semantics")
def edit_table_semantics(db: str, table: str, request: SemanticsEdit):
    """
    One field corrected in the studio (the description of the table or of a column, data_as_of, categorical),
    or the descriptions the LLM rewrote marked read.
    """
    _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Unknown table: {table}")
    order = [c["name"] for c in listing["columns"].get(table) or []]
    if request.field in ("column", "categorical") and order and request.column not in order:
        raise HTTPException(400, f"Unknown column: {table}.{request.column}")
    try:
        _update_semantics(db, request.version, lambda semantics: {
            **semantics, table: state.edit_semantics(semantics.get(table), request.field, request.value,
                                                     request.column, order)})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    logger.info(f"[studio] {db}.{table}: {request.field} {request.column or ''} edited")
    return table_semantics(db, table)


def _start(db: str, step: str, command: list[str], tables: list[str] | None):
    if unknown := sorted(set(tables or []) - set(_listing(db)["tables"])):
        raise HTTPException(400, f"Unknown table(s): {', '.join(unknown)}")
    try:
        return runner.start(db, step, command, tables).to_dict()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


class ProfileRequest(BaseModel):
    tables: list[str] | None = None
    missing: bool = False


@app.post("/api/db/{db}/profile")
def profile(db: str, request: ProfileRequest):
    """Profile the tables chosen, or the ones missing from the profile (make profile ARGS=...)."""
    _db_config(db)
    if not request.tables and not request.missing:
        raise HTTPException(400, "Choose tables, or the missing ones")
    return _start(db, "profile", profile_command(db, request.tables, request.missing), request.tables)


class SemanticsRequest(BaseModel):
    tables: list[str] | None = None
    fresh: bool = False
    replace_edited: bool = False


@app.post("/api/db/{db}/semantics")
def describe(db: str, request: SemanticsRequest):
    """
    Describe the tables missing a description, or describe the tables chosen again (make semantics ARGS=...),
    the descriptions corrected by hand kept unless replace_edited.
    """
    _db_config(db)
    if request.fresh and not request.tables:
        raise HTTPException(400, "Choose the tables to describe again")
    command = semantics_command(db, request.tables, request.fresh, request.replace_edited)
    return _start(db, "semantics", command, request.tables)


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


# ------------------------------------------------------------
# Relations: the map of the tables, the evidence of each relation, the path of the agent
# ------------------------------------------------------------

def _relations_version(db: str) -> str:
    """Version of relationships.yaml (its modification time), as _version for tables.yaml."""
    path = relations_path(db)
    return str(path.stat().st_mtime_ns if path.exists() else 0)


def _update_relations(db: str, version: str, change) -> None:
    """Rewrite relationships.yaml with change(entry), unless it changed since the page read it."""
    with _lock:
        if version != _relations_version(db):
            raise HTTPException(409, "relationships.yaml changed since the page read it: reload the page")
        write_relations_file(db, change(load_relations_file(db)))


@app.get("/api/db/{db}/relations")
def relations(db: str):
    """The Relations screen: the relations with their evidence, the ignored ones, the map of the tables."""
    _db_config(db)
    listing = _listing(db)
    view = relations_view(listing["tables"], load_relations_file(db), load_profile(db))
    columns = {table: [c["name"] for c in cols] for table, cols in listing["columns"].items()}
    return {**view, "columns": columns, "version": _relations_version(db)}


@app.get("/api/db/{db}/relations/path")
def relations_path_between(db: str, a: str, b: str):
    """The join path the agent takes between two tables (lib/fk_graph.py)."""
    _db_config(db)
    tables = _listing(db)["tables"]
    if a not in tables or b not in tables:
        raise HTTPException(404, f"Unknown table: {a if a not in tables else b}")
    return join_path(a, b, fk_graph(load_relations_file(db)["relations"], tables))


def _column_ref(db: str, ref: str) -> str:
    """A table.column of the database, or 400."""
    table, _, column = ref.partition(".")
    columns = [c["name"] for c in _listing(db)["columns"].get(table) or []]
    if column not in columns:
        raise HTTPException(400, f"Unknown column: {ref}")
    return ref


class RelationRequest(BaseModel):
    source: str
    target: str
    version: str
    force: bool = False


@app.post("/api/db/{db}/relations")
def add(db: str, request: RelationRequest):
    """
    A relation added from the page, measured on the data first: the share of its values found, and whether
    the column it points to is a key. When it is not, a join would count each row of the source several
    times: the relation is not added unless forced (`added` false, the measures in `check`).
    """
    _db_config(db)
    source, target = _column_ref(db, request.source), _column_ref(db, request.target)
    if source.split(".")[0] == target.split(".")[0]:
        raise HTTPException(400, "Two columns of two different tables")
    if db not in _databases:
        raise HTTPException(503, "The database does not answer: the relation can't be checked")
    try:
        found  = key_overlap(_databases[db], source, target)
        spread = key_spread(_databases[db], source, target)
    except Exception as e:
        raise HTTPException(502, f"The check of the relation failed: {str(e).splitlines()[0][:200]}") from e
    found = None if found is None else round(found, 3)
    check = {"source": source, "target": target, "found": found, **spread}
    if spread["per_value"][1] > 1 and not request.force:
        logger.info(f"[studio] {db}: relation {source} -> {target} not added, {target} is not a key")
        return {**relations(db), "added": False, "check": check}
    _update_relations(db, request.version, lambda entry: add_relation(entry, source, target, found, spread))
    logger.info(f"[studio] {db}: relation {source} -> {target} added ({found})")
    return {**relations(db), "added": True, "check": check}


@app.post("/api/db/{db}/relations/ignore")
def ignore(db: str, request: RelationRequest, back: bool = False):
    """A relation moved to `ignore` (it never comes back), or back from it (back=true)."""
    _db_config(db)
    try:
        _update_relations(db, request.version,
                          lambda entry: ignore_relation(entry, request.source, request.target, ignore=not back))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    logger.info(f"[studio] {db}: relation {request.source} -> {request.target} {'restored' if back else 'ignored'}")
    return relations(db)


class RelationsRun(BaseModel):
    check: bool = False


@app.post("/api/db/{db}/relations/run")
def find_relations(db: str, request: RelationsRun):
    """Find the relations (measured, then LLM), or only check the declared ones again (make relations ARGS=...)."""
    _db_config(db)
    return _start(db, "relations", relations_command(db, request.check), None)


# ------------------------------------------------------------
# Metrics: each metric checked against the rules of the agent, tested on the data (counts, never its value)
# ------------------------------------------------------------

_tests: dict[str, dict[str, dict]] = {}   # {db: {metric: {key, result}}}: the last test of each, while it is unchanged
_runs:  dict[str, dict] = {}              # {db: what the last run of the generator skipped and added}


def _metrics_version(db: str) -> str:
    """Version of metrics.yaml (its modification time), as _version for tables.yaml."""
    path = metrics_path(db)
    return str(path.stat().st_mtime_ns if path.exists() else 0)


def _update_metrics(db: str, version: str, change) -> None:
    """Rewrite metrics.yaml with change(entries, ignored), unless it changed since the page read it."""
    with _lock:
        if version != _metrics_version(db):
            raise HTTPException(409, "metrics.yaml changed since the page read it: reload the page")
        write_metrics_file(db, *change(*read_metrics_file(db)))


def _metrics_context(db: str) -> dict:
    """What the checks of a metric read: the columns of the database and their types, the hidden ones, the formats, the relations."""
    db_cfg    = _db_config(db)
    listing   = _listing(db)
    semantics = load_semantics(db)
    return {
        "columns":      {table: [c["name"] for c in cols] for table, cols in listing["columns"].items()},
        "column_types": {f"{table}.{c['name']}": c["type"] for table, cols in listing["columns"].items() for c in cols},
        "hidden":       {table: excluded_columns(entry) for table, entry in semantics.items()},
        "formats":      {table: (entry or {}).get("formats") or {} for table, entry in semantics.items()},
        "fk_relations": load_fk_relations(db),
        "dialect":      db_cfg.dialect,
    }


def _last_run(db: str) -> dict:
    """What the last run of the generator on the database skipped and added, read from its lines."""
    job = runner.job
    if job and job.db == db and job.step == "metrics":
        _runs[db] = last_run(job.lines)
    return _runs.get(db) or {}


def _metric(db: str, name: str, entries: dict, ignored: list[str], context: dict) -> dict:
    if name not in entries:
        raise HTTPException(404, f"Unknown metric: {name}")
    return metric_row(name, entries[name], entries, ignored, **context)


def _database(db: str):
    if db not in _databases:
        raise HTTPException(503, "The database does not answer: the metric can't be checked")
    return _databases[db]


def _first_line(error: Exception) -> str:
    return (str(error).splitlines() or [""])[0][:300]


@app.get("/api/db/{db}/metrics")
def metrics_screen(db: str):
    """The Metrics screen: each metric with its checks and its last test, the ignored names, the last run."""
    context = _metrics_context(db)
    entries, ignored = read_metrics_file(db)
    view = metrics_view(entries, ignored, tests=_tests.get(db, {}), run=_last_run(db), **context)
    return {**view, "dialect": context["dialect"], "version": _metrics_version(db)}


@app.post("/api/db/{db}/metrics/run")
def propose_metrics(db: str):
    """Propose metrics with the LLM (make metrics): added to metrics.yaml, a known or rejected one never again."""
    _db_config(db)
    return _start(db, "metrics", metrics_command(db), None)


@app.post("/api/db/{db}/metrics/{name}/test")
def test_one_metric(db: str, name: str):
    """
    A metric tested on the data, counts only: the rows its joins count, its measures and its date
    filled, the sign of its result, never its value. Kept until the metric changes.
    """
    context = _metrics_context(db)
    row = _metric(db, name, *read_metrics_file(db), context)
    if not row["sql"]:
        raise HTTPException(400, f"The metric can't be computed: {row['error']['detail']}")
    database = _database(db)
    plan     = measure_queries(row, context["column_types"], context["fk_relations"], context["dialect"])
    started  = time.time()
    try:
        with database._engine.connect() as conn:
            main = tuple(conn.execute(text(plan["main"])).fetchone())
            rows = {table: conn.execute(text(sql)).scalar() for table, sql in plan["tables"].items()}
        result = read_measures(plan, main, rows, time.time() - started)
    except Exception as e:
        result = {"error": _first_line(e), "seconds": round(time.time() - started, 1)}
    result["at"] = time.time()
    _tests.setdefault(db, {})[name] = {"key": measure_key(row), "result": result}
    logger.info(f"[studio] {db}: metric {name} tested ({result.get('error') or 'counts read'})")
    return metrics_screen(db)


class MetricEdit(BaseModel):
    field: Literal["description", "synonyms", "expr", "time_dimension"]
    value: str | None = None
    version: str


@app.post("/api/db/{db}/metrics/{name}")
def edit_one_metric(db: str, name: str, request: MetricEdit):
    """
    One field of a metric corrected in the studio. A new expression or date is checked first: the
    startup validation of the agent, its joins, then the database compiles it on an empty result.
    """
    context = _metrics_context(db)
    entries, ignored = read_metrics_file(db)
    change = lambda entries, ignored: (edit_metric(entries, name, request.field, request.value), ignored)  # noqa: E731
    try:
        edited, _ = change(entries, ignored)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if request.field in ("expr", "time_dimension"):
        row = _metric(db, name, edited, ignored, context)
        if row["error"]:
            detail = row["error"]["detail"]
            raise HTTPException(400, f"reads a column hidden from the agent: {detail}" if row["error"]["kind"] == "hidden" else detail)
        database = _database(db)
        try:
            with database._engine.connect() as conn:
                conn.execute(text(f"{row['sql']}\nWHERE 1 = 0"))
        except Exception as e:
            raise HTTPException(400, f"The database refuses it: {_first_line(e)}") from e
    _update_metrics(db, request.version, change)
    logger.info(f"[studio] {db}: metric {name} {request.field} edited")
    return metrics_screen(db)


class MetricVersion(BaseModel):
    version: str


@app.post("/api/db/{db}/metrics/{name}/ignore")
def ignore_one_metric(db: str, name: str, request: MetricVersion, back: bool = False):
    """A metric rejected (deleted, its name in `ignore`: never proposed again), or its name out of `ignore` (back=true)."""
    _db_config(db)
    try:
        _update_metrics(db, request.version, lambda entries, ignored: ignore_metric(entries, ignored, name, ignore=not back))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _tests.get(db, {}).pop(name, None)
    logger.info(f"[studio] {db}: metric {name} {'allowed again' if back else 'ignored'}")
    return metrics_screen(db)


# ------------------------------------------------------------
# Index: what the agent reads of each table against the files today, and the eval of the agent
# ------------------------------------------------------------

def _eval_results(db: str) -> tuple[dict | None, float | None]:
    """The results of the last eval of the database (make eval) and the time of their file, or (None, None)."""
    path = RESULTS_DIR / f"{db.lower()}.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), path.stat().st_mtime


@app.get("/api/db/{db}/index")
def index_screen(db: str):
    """
    The Index screen: each table with its status in the index, what the agent receives, the last eval.
    The dates of the files and of the value index tell how old an index built before its trace is.
    """
    db_cfg  = _db_config(db)
    listing = _listing(db)
    columns = {table: [c["name"] for c in cols] for table, cols in listing["columns"].items()}
    view    = index_view(listing["tables"], columns, load_semantics(db), load_fk_relations(db), read_trace(db),
                         read_metrics_file(db)[0], db_cfg.dialect, eval_run(*_eval_results(db)))
    files   = {name: path.stat().st_mtime for name, path in (("tables.yaml", tables_path(db)),
                                                             ("relationships.yaml", relations_path(db))) if path.exists()}
    values  = value_index_path(db)
    return {**view, "files": files, "values_at": values.stat().st_mtime if values.exists() else None,
            "sample_rows_now": db_cfg.sample_rows, "dataset": (DATASETS_DIR / f"{db.lower()}.yaml").exists()}


@app.get("/api/db/{db}/index/{table}")
def index_document(db: str, table: str):
    """The document of a table as the agent reads it, and what changed since the index."""
    _db_config(db)
    listing, trace = _listing(db), read_trace(db)
    if table not in listing["tables"] and table not in (trace or {}).get("tables", {}):
        raise HTTPException(404, f"Unknown table: {table}")
    view = document_view(table, trace, load_semantics(db).get(table), load_fk_relations(db), listing["columns"].get(table))
    return {**view, "gone": listing["source"] == "database" and table not in listing["tables"]}


@app.post("/api/db/{db}/index/run")
def run_index(db: str):
    """Index the database again (make index): the documents of the tables, the value index and its trace."""
    _db_config(db)
    return _start(db, "index", index_command(db), None)


@app.post("/api/db/{db}/eval/run")
def run_eval(db: str):
    """Ask the questions of the eval to the agent (make eval): offline/eval/results/<db>.json."""
    _db_config(db)
    if not (DATASETS_DIR / f"{db.lower()}.yaml").exists():
        raise HTTPException(400, f"No eval questions for {db}: offline/eval/datasets/{db.lower()}.yaml")
    return _start(db, "eval", eval_command(db), None)


@app.get("/eval/{db}")
def eval_dashboard(db: str):
    """
    The eval dashboard of this database alone, built from its results at each request (make eval-dashboard
    builds every database into one file): the results of a database stay out of the page of another.
    """
    _db_config(db)
    path = RESULTS_DIR / f"{db.lower()}.json"
    if not path.exists():
        raise HTTPException(404, f"{db} was never evaluated: make eval DB={db}")
    with tempfile.TemporaryDirectory() as folder:
        html = build_dashboard([path], Path(folder) / "dashboard.html").read_text()
    return HTMLResponse(html)
