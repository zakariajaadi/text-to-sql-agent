# offline/studio/app.py

"""
Studio of the semantic layer: a web page to run the pipeline steps and see what they write, instead
of patch scripts (Profiling, Semantics, Relations, Metrics, then Index: what the agent reads, and its eval).
The home shows every database at once from its files (what the agent reads, the step to do next) and
searches through their semantic layer. The Périmètre: every table of the database against the tables chosen
for the agent and the pipeline (scope.yaml) and the last index. Step 1, profiling: the tables and their status at each step, profile the ones chosen
or missing, read the profile of a table (declared type against what the data holds), make an exception
to what it detects, hide the personal data of the whole database.

    PYTHONPATH=src python -m uvicorn offline.studio.app:app --port 8100     (see `make studio`)

The YAML files of semantic_layer/<db>/ stay the source of truth: the CLI and the Makefile keep working,
the studio reads the files at each request and writes tables.yaml only on a click of the page.
It shows shapes and counts of the data, never a value. The database is read to list the tables, to
measure a relation added and to test a metric (counts only). Its messages are in French, for the page
to show as they are.
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
from lib.semantics import (
    excluded_columns,
    load_fk_relations,
    load_profile,
    load_scope,
    load_semantics,
    scope_origin,
    scope_path,
    tables_path,
)
from lib.value_index import value_index_path
from offline.eval.dashboard import DATASETS_DIR, RESULTS_DIR, build_dashboard
from offline.eval.history import read_history
from offline.indexer.trace import read_trace
from offline.semantic_gen.profiling import PARTITIONED, key_overlap, key_spread
from offline.semantic_gen.utils import (
    dump_table_semantics,
    load_relations_file,
    relations_path,
    write_metrics_file,
    write_relations_file,
    write_scope,
)
from offline.studio import state
from offline.studio.home import search_layer
from offline.studio.index import document_text, document_view, eval_run, index_view
from offline.studio.reading import edit_read_mode, reading
from offline.studio.scope import change_scope, scope_rows, scope_view
from offline.studio.table_types import table_types
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
    keep_relation,
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
        raise HTTPException(404, f"Base inconnue : {db}")
    return cfg.get_db(db)


def _listing(db: str, refresh: bool = False) -> dict:
    """
    The tables of the scope the database has (`tables`, lib/semantics.load_scope) and their columns, and every
    table of the database (`base`, for the Périmètre tab). When the database can't be reached, the tables of the
    scope, else the ones known from the files (profile.yaml, tables.yaml), `base` None, with the error to show.
    """
    with _lock:
        if not refresh and db in _listings:
            return _listings[db]
        scope = load_scope(db)
        try:
            if db not in _databases:
                _databases[db] = build_database(_db_config(db))
            inspector = sa_inspect(_databases[db]._engine)
            base      = sorted(inspector.get_table_names())
            tables    = [t for t in base if scope is None or t in scope]
            listing   = {"tables": tables, "base": base, "source": "database", "error": None,
                         "columns": {t: _columns(inspector, t) for t in tables}}
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[studio] {db}: tables read from the files, the database is unreachable — {e}")
            _databases.pop(db, None)
            known   = scope if scope is not None else list(dict.fromkeys([*load_profile(db), *load_semantics(db)]))
            listing = {"tables": known, "base": None, "columns": {}, "source": "files", "error": str(e).splitlines()[0][:300]}
        listing["read_at"] = time.time()
        _listings[db] = listing
        return listing


def _columns(inspector, table: str) -> list[dict]:
    return [{"name": c["name"], "type": _type_name(c["type"])} for c in inspector.get_columns(table)]


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


def _known_listing(db: str) -> dict:
    """
    The tables of a database without asking it (the home screen reads every database at once, and Impala
    can be slow or unreachable): the last listing read, else the tables the files and the last index know.
    """
    with _lock:
        if db in _listings:
            return _listings[db]
    if (scope := load_scope(db)) is not None:
        return {"tables": scope, "base": None, "columns": {}, "source": "files", "error": None}
    indexed = (read_trace(db) or {}).get("tables", {})
    known   = list(dict.fromkeys([*load_profile(db), *load_semantics(db), *indexed]))
    return {"tables": known, "base": None, "columns": {}, "source": "files", "error": None}


def _table_types(db: str, listing: dict, profile: dict, semantics: dict, relations) -> dict[str, dict]:
    """The type of each table, its columns typed as the database lists them, else as the last index read them."""
    columns = {table: {c["name"]: c["type"] for c in cols} for table, cols in listing["columns"].items()}
    indexed = (read_trace(db) or {}).get("tables", {})
    for table in listing["tables"]:
        if not columns.get(table) and table in indexed:
            columns[table] = indexed[table]["columns"]
    return table_types(listing["tables"], columns, profile, semantics, relations)


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
    return [{"name": d.name, "dialect": d.dialect, "personal": d.exclude_personal_data, "tagline": d.tagline}
            for d in cfg.databases]


@app.get("/api/overview")
def overview():
    """
    The home screen: for each database, the counts of each step and what the agent reads (the Index screen),
    from the files only: no database is asked, the page opens even when one of them is unreachable.
    """
    bases = []
    for db in [d.name for d in cfg.databases]:
        db_cfg    = _db_config(db)
        listing   = _known_listing(db)
        relations = load_fk_relations(db)
        metrics   = _load_metrics(db)
        names = {table: [c["name"] for c in columns] for table, columns in listing["columns"].items()}
        rows  = state.table_rows(listing["tables"], names, load_profile(db), load_semantics(db), relations, metrics,
                                 db_cfg.exclude_personal_data)
        bases.append({"name": db, "dialect": db_cfg.dialect, "source": listing["source"],
                      "summary": state.summary(rows, relations, metrics), "index": _index_payload(db, listing)})
    return bases


@app.get("/api/search")
def search(q: str = ""):
    """The tables and columns of every database whose name or description holds `q` (the home screen)."""
    return search_layer(q, {d.name: (load_semantics(d.name), read_trace(d.name)) for d in cfg.databases})


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
    types = _table_types(db, listing, profile, semantics, relations)
    partitioned = db_cfg.dialect in PARTITIONED
    for row in rows:
        facts = profile.get(row["name"])
        read  = reading(semantics.get(row["name"]), facts, partitioned_engine=partitioned)
        row["type"], row["type_edited"] = types[row["name"]]["type"], types[row["name"]]["edited"]
        row["read_warn"] = read["warn"]
        # What an older profiler left out of the profile of the table, for "Compléter le profil"
        row["outdated"] = [*state.outdated(facts, names.get(row["name"]) or []), *(["partitions"] if read["unread"] else [])] \
            if facts else []
    found    = state.personal_columns(profile, semantics)
    personal = [s for columns in found.values() for s in columns.values()]
    return {
        "db": db, "source": listing["source"], "error": listing["error"], "read_at": listing["read_at"],
        "version": _version(db), "tables": rows,
        "personal": {**{visibility: personal.count(visibility) for visibility in ("hidden", "kept", "visible")},
                     "hidden_tables": sum("hidden" in columns.values() for columns in found.values())},
        "summary": state.summary(rows, relations, metrics),
    }


@app.get("/api/db/{db}/tables/{table}")
def table_detail(db: str, table: str):
    """The profile of a table and its proposals, for the Profilage card, with the properties of the table (_properties)."""
    db_cfg  = _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Table inconnue : {table}")
    db_columns = listing["columns"].get(table)
    profile, semantics = load_profile(db), load_semantics(db)
    detail = state.table_detail(table, profile.get(table), semantics.get(table), db_cfg.exclude_personal_data, db_columns)
    return {**detail, "columns": len(db_columns) if db_columns else None, **_properties(db, table), "version": _version(db)}


def _properties(db: str, table: str) -> dict:
    """
    What a table is and how to read it, shown and chosen in the Profilage card: its type (detected or corrected,
    offline/studio/table_types.py), its read mode checked against its columns (offline/studio/reading.py) and
    the date its data stops.
    """
    listing, profile, semantics = _listing(db), load_profile(db), load_semantics(db)
    columns = [c["name"] for c in listing["columns"].get(table) or []]
    return {
        "table_type": _table_types(db, listing, profile, semantics, load_fk_relations(db))[table],
        "reading":    reading(semantics.get(table), profile.get(table), columns, _db_config(db).dialect in PARTITIONED),
        "data_as_of": str((semantics.get(table) or {}).get("data_as_of") or ""),
    }


@app.get("/api/db/{db}/tables/{table}/semantics")
def table_semantics(db: str, table: str):
    """A table as the Descriptions screen shows it: the descriptions of the table and of its columns, categorical columns."""
    _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Table inconnue : {table}")
    semantics, profile = load_semantics(db), load_profile(db)
    detail = state.semantics_detail(table, semantics.get(table), profile.get(table), listing["columns"].get(table))
    return {**detail, "version": _version(db)}


class SemanticsChange(BaseModel):
    field: Literal["description", "column", "categorical", "unclear", "read"]
    value: str | bool | None = None
    column: str | None = None


class SemanticsEdit(BaseModel):
    changes: list[SemanticsChange]
    version: str


@app.post("/api/db/{db}/tables/{table}/semantics")
def edit_table_semantics(db: str, table: str, request: SemanticsEdit):
    """
    The draft of the Descriptions card saved (the description of the table, of its columns, categorical, unclear),
    written at once or not at all; or the descriptions the LLM rewrote marked read.
    """
    _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Table inconnue : {table}")
    order = [c["name"] for c in listing["columns"].get(table) or []]
    for change in request.changes:
        if change.field in ("column", "categorical", "unclear") and order and change.column not in order:
            raise HTTPException(400, f"Colonne inconnue : {table}.{change.column}")

    def edit(entry: dict | None) -> dict:
        for change in request.changes:
            entry = state.edit_semantics(entry, change.field, change.value, change.column, order)
        return entry

    try:
        _update_semantics(db, request.version, lambda semantics: {**semantics, table: edit(semantics.get(table))})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    logger.info(f"[studio] {db}.{table}: {', '.join(f'{c.field} {c.column or ""}'.strip() for c in request.changes)} edited")
    return table_semantics(db, table)


class ReadMode(BaseModel):
    mode: Literal["last", "all", "live"] | None = None
    partition_column: str | None = None
    key_columns: list[str] = []
    version_column: str | None = None


class PropertiesEdit(BaseModel):
    table_type: str | None = None   # empty: the detection again
    data_as_of: str | None = None   # YYYY-MM-DD, empty: none
    read_mode: ReadMode | None = None
    version: str


@app.post("/api/db/{db}/tables/{table}/properties")
def edit_table_properties(db: str, table: str, request: PropertiesEdit):
    """
    The properties of a table chosen in the Profilage card, the ones sent written at once or not at all: its type,
    the date its data stops, its read mode (offline/studio/reading.py; mode None: the table read as it is). Returns
    the properties of the table (the type follows the mode) and the version of tables.yaml.
    """
    _db_config(db)
    listing = _listing(db)
    if table not in listing["tables"]:
        raise HTTPException(404, f"Table inconnue : {table}")
    columns = [c["name"] for c in listing["columns"].get(table) or []]
    sent = request.model_fields_set - {"version"}

    def edit(entry: dict | None) -> dict:
        for field in ("table_type", "data_as_of"):
            if field in sent:
                entry = state.edit_semantics(entry, field, getattr(request, field))
        if "read_mode" in sent:
            mode = request.read_mode or ReadMode()
            entry = edit_read_mode(entry, mode.mode, mode.partition_column, mode.key_columns, mode.version_column, columns)
        return entry

    try:
        _update_semantics(db, request.version, lambda semantics: {**semantics, table: edit(semantics.get(table))})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    logger.info(f"[studio] {db}.{table}: {', '.join(sorted(sent))} edited")
    return {**_properties(db, table), "version": _version(db)}


def _start(db: str, step: str, command: list[str], tables: list[str] | None):
    if unknown := sorted(set(tables or []) - set(_listing(db)["tables"])):
        raise HTTPException(400, f"{'Tables inconnues' if len(unknown) > 1 else 'Table inconnue'} : {', '.join(unknown)}")
    try:
        return runner.start(db, step, command, tables).to_dict()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


class ProfileRequest(BaseModel):
    tables: list[str] | None = None
    missing: bool = False
    complete: bool = False   # only what an older profile lacks, of `tables` or of every table


@app.post("/api/db/{db}/profile")
def profile(db: str, request: ProfileRequest):
    """
    Profile the tables chosen, or the ones missing from the profile; or complete an older profile of the
    tables chosen, or of all of them (make profile ARGS=...).
    """
    _db_config(db)
    if not request.tables and not request.missing and not request.complete:
        raise HTTPException(400, "Coche des tables, ou choisis les manquantes")
    return _start(db, "profile", profile_command(db, request.tables, request.missing, request.complete), request.tables)


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
        raise HTTPException(400, "Coche les tables à redécrire")
    command = semantics_command(db, request.tables, request.fresh, request.replace_edited)
    return _start(db, "semantics", command, request.tables)


def _update_semantics(db: str, version: str, change) -> None:
    """
    Rewrite tables.yaml with change(semantics), unless it changed since the page read it (`version`).
    Written at once (a temporary file renamed): a run of the CLI may read it at the same time.
    """
    with _lock:
        if version != _version(db):
            raise HTTPException(409, "tables.yaml a changé depuis l'ouverture de la page : rouvre la table")
        semantics = change(load_semantics(db))
        path = tables_path(db)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(dump_table_semantics(semantics))
        temporary.replace(path)


def _profiled(db: str, table: str) -> dict:
    _db_config(db)
    if not (facts := load_profile(db).get(table)):
        raise HTTPException(404, f"{table} n'est pas encore profilée")
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
            raise HTTPException(400, f"Le profil ne détecte aucun format dans {table}.{column}")
        change = lambda entry: state.set_format(entry, column, value if request.action == "format" else None)  # noqa: E731
    else:
        if column not in (facts.get("exclude") or {}):
            raise HTTPException(400, f"Le profil ne propose pas de cacher {table}.{column}")
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
            raise HTTPException(409, "relationships.yaml a changé depuis l'ouverture de la page : recharge la page")
        write_relations_file(db, change(load_relations_file(db)))


@app.get("/api/db/{db}/relations")
def relations(db: str):
    """The Relations screen: the relations with their evidence, the ignored ones, the map of the tables, and the
    tables typed as references (the map writes them in pills on the tables joining them, not as boxes)."""
    _db_config(db)
    listing = _listing(db)
    profile = load_profile(db)
    view = relations_view(listing["tables"], load_relations_file(db), profile)
    columns = {table: [c["name"] for c in cols] for table, cols in listing["columns"].items()}
    types = _table_types(db, listing, profile, load_semantics(db), load_fk_relations(db))
    references = [t for t in listing["tables"] if types[t]["type"] == "reference"]
    return {**view, "columns": columns, "references": references, "version": _relations_version(db)}


@app.get("/api/db/{db}/relations/path")
def relations_path_between(db: str, a: str, b: str):
    """The join path the agent takes between two tables (lib/fk_graph.py)."""
    _db_config(db)
    tables = _listing(db)["tables"]
    if a not in tables or b not in tables:
        raise HTTPException(404, f"Table inconnue : {a if a not in tables else b}")
    return join_path(a, b, fk_graph(load_relations_file(db)["relations"], tables))


def _column_ref(db: str, ref: str) -> str:
    """A table.column of the database, or 400."""
    table, _, column = ref.partition(".")
    columns = [c["name"] for c in _listing(db)["columns"].get(table) or []]
    if column not in columns:
        raise HTTPException(400, f"Colonne inconnue : {ref}")
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
        raise HTTPException(400, "Choisis deux colonnes de deux tables différentes")
    if db not in _databases:
        raise HTTPException(503, "La base ne répond pas : la jointure ne peut pas être vérifiée")
    try:
        found  = key_overlap(_databases[db], source, target)
        spread = key_spread(_databases[db], source, target)
    except Exception as e:
        raise HTTPException(502, f"La vérification de la jointure a échoué : {str(e).splitlines()[0][:200]}") from e
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


@app.post("/api/db/{db}/relations/keep")
def keep(db: str, request: RelationRequest, back: bool = False):
    """A weak relation kept by a human (kept: true, no longer in amber), or no longer kept (back=true)."""
    _db_config(db)
    try:
        _update_relations(db, request.version,
                          lambda entry: keep_relation(entry, request.source, request.target, keep=not back))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    logger.info(f"[studio] {db}: relation {request.source} -> {request.target} {'no longer kept' if back else 'kept'}")
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
            raise HTTPException(409, "metrics.yaml a changé depuis l'ouverture de la page : recharge la page")
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
        raise HTTPException(404, f"Métrique inconnue : {name}")
    return metric_row(name, entries[name], entries, ignored, **context)


def _database(db: str):
    if db not in _databases:
        raise HTTPException(503, "La base ne répond pas : la métrique ne peut pas être vérifiée")
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
        raise HTTPException(400, f"La métrique ne peut pas être calculée : {row['error']['detail']}")
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
    changes: dict[Literal["description", "synonyms", "expr", "time_dimension"], str | None]   # synonyms: comma-separated
    version: str


@app.post("/api/db/{db}/metrics/{name}")
def edit_one_metric(db: str, name: str, request: MetricEdit):
    """
    The draft of a metric saved from the studio: its fields written at once, or none of them. A new expression
    or date is checked first: the startup validation of the agent, its joins, then the database compiles it on
    an empty result.
    """
    context = _metrics_context(db)
    entries, ignored = read_metrics_file(db)

    def change(entries: dict, ignored: list) -> tuple[dict, list]:
        for field, value in request.changes.items():
            entries = edit_metric(entries, name, field, value)
        return entries, ignored

    try:
        edited, _ = change(entries, ignored)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if {"expr", "time_dimension"} & request.changes.keys():
        row = _metric(db, name, edited, ignored, context)
        if row["error"]:
            detail = row["error"]["detail"]
            raise HTTPException(400, f"La métrique lit une colonne cachée à l'agent : {detail}" if row["error"]["kind"] == "hidden" else detail)
        database = _database(db)
        try:
            with database._engine.connect() as conn:
                conn.execute(text(f"{row['sql']}\nWHERE 1 = 0"))
        except Exception as e:
            raise HTTPException(400, f"La base refuse la métrique : {_first_line(e)}") from e
    _update_metrics(db, request.version, change)
    logger.info(f"[studio] {db}: metric {name} {', '.join(request.changes)} edited")
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
# Périmètre: every table of the database against the scope (scope.yaml) and the last index
# ------------------------------------------------------------

def _scope_version(db: str) -> str:
    """Version of scope.yaml (its modification time), as _version for tables.yaml."""
    path = scope_path(db)
    return str(path.stat().st_mtime_ns if path.exists() else 0)


def _scope_payload(db: str, listing: dict) -> dict:
    """The Périmètre tab: each table of the database, of the scope or of the index, with its state."""
    db_cfg    = _db_config(db)
    scope     = load_scope(db)
    trace     = read_trace(db)
    semantics = load_semantics(db)
    columns   = {table: [c["name"] for c in cols] for table, cols in listing["columns"].items()}
    view      = index_view(listing["tables"], columns, semantics, load_fk_relations(db), trace, {}, db_cfg.dialect, None,
                           listing["base"])
    rows = scope_rows(listing["base"], scope, {row["name"]: row["status"] for row in view["tables"]},
                      set((trace or {}).get("tables", {})), set(load_profile(db)),
                      {table for table, entry in semantics.items() if (entry or {}).get("description")},
                      {table: len(cols) for table, cols in listing["columns"].items()})
    return {**scope_view(rows, listing["base"], scope_origin(db)),
            "source": listing["source"], "error": listing["error"], "read_at": listing["read_at"],
            "trace": trace is not None, "version": _scope_version(db)}


@app.get("/api/db/{db}/scope")
def scope_screen(db: str):
    """The Périmètre tab of a database, its tables read in the database."""
    _db_config(db)
    return _scope_payload(db, _listing(db))


class ScopeChange(BaseModel):
    add: list[str] = []
    remove: list[str] = []
    all: bool = False   # every table, the ones to come too
    version: str


@app.post("/api/db/{db}/scope")
def edit_scope(db: str, request: ScopeChange):
    """
    Every table, the ones to come too (`all`), or tables added to the scope or taken out of it: a list, written to
    scope.yaml (with neither: the scope as it is today, every table without scope.yaml, written as a list). The
    tables entering it get their columns read first: nothing is written when the database can't give them. The
    other screens read the new scope at once; the agent and the index at their next start.
    """
    _db_config(db)
    listing = _listing(db)
    if listing["base"] is None:
        raise HTTPException(503, "La base ne répond pas : le périmètre se choisit sur la liste de ses tables")
    with _lock:
        if request.version != _scope_version(db):
            raise HTTPException(409, "scope.yaml a changé depuis l'ouverture de la page : recharge la page")
        if request.all and (request.add or request.remove):
            raise HTTPException(400, "Toutes les tables, ou des tables ajoutées et retirées : pas les deux")
        try:
            scope = None if request.all else change_scope(load_scope(db), listing["base"], request.add, request.remove)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        tables = [t for t in listing["base"] if scope is None or t in scope]
        try:
            inspector = sa_inspect(_databases[db]._engine)
            columns = {t: listing["columns"].get(t) or _columns(inspector, t) for t in tables}
        except Exception as e:
            raise HTTPException(502, f"La lecture des colonnes a échoué : {_first_line(e)}") from e
        write_scope(db, scope)
        _listings[db] = {**listing, "tables": tables, "columns": columns}
    logger.info(f"[studio] {db}: scope " + ("every table" if scope is None else
                                            f"{len(scope)} tables (+{len(request.add)}, -{len(request.remove)})"))
    return scope_screen(db)


# ------------------------------------------------------------
# Index: what the agent reads of each table against the files today, and the eval of the agent
# ------------------------------------------------------------

def _questions(db: str) -> int | None:
    """The number of eval questions of the database (offline/eval/datasets/<db>.yaml), None without a dataset."""
    try:
        dataset = yaml.safe_load((DATASETS_DIR / f"{db.lower()}.yaml").read_text())
        questions = dataset["questions"] if isinstance(dataset, dict) else dataset
        return len(questions) if isinstance(questions, list) else None
    except Exception:
        return None


def _eval_results(db: str) -> tuple[dict | None, float | None]:
    """The results of the last eval of the database (make eval) and the time of their file, or (None, None)."""
    path = RESULTS_DIR / f"{db.lower()}.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), path.stat().st_mtime


def _index_payload(db: str, listing: dict) -> dict:
    """
    What the Index screen shows of a database, its tables as `listing` gives them: each table with its status
    in the index, what the agent receives, the last eval and the questions of the next one. The dates of the
    files and of the value index tell how old an index built before its trace is. `scope`: the tables of the
    scope against the ones of the database (None when it can't be read), for the tile of the Vue d'ensemble.
    """
    db_cfg  = _db_config(db)
    columns = {table: [c["name"] for c in cols] for table, cols in listing["columns"].items()}
    rows    = {table: (facts or {}).get("rows") for table, facts in load_profile(db).items()}
    history = read_history(RESULTS_DIR / f"{db.lower()}.json")
    view    = index_view(listing["tables"], columns, load_semantics(db), load_fk_relations(db), read_trace(db),
                         read_metrics_file(db)[0], db_cfg.dialect, eval_run(*_eval_results(db), history), listing["base"], rows)
    files   = {name: path.stat().st_mtime for name, path in (("tables.yaml", tables_path(db)),
                                                             ("relationships.yaml", relations_path(db))) if path.exists()}
    values  = value_index_path(db)
    return {**view, "files": files, "values_at": values.stat().st_mtime if values.exists() else None,
            "sample_rows_now": db_cfg.sample_rows, "dataset": (DATASETS_DIR / f"{db.lower()}.yaml").exists(),
            "questions": _questions(db),
            "scope": {"tables": len(listing["tables"]), "base": None if listing["base"] is None else len(listing["base"]),
                      "origin": scope_origin(db), "version": _scope_version(db)}}


@app.get("/api/db/{db}/index")
def index_screen(db: str):
    """The Index screen of a database, its tables read in the database."""
    _db_config(db)
    return _index_payload(db, _listing(db))


@app.get("/api/db/{db}/index/{table}")
def index_document(db: str, table: str):
    """The document of a table as the agent reads it (its columns, its text, the data masked), and what changed since the index."""
    db_cfg = _db_config(db)
    listing, trace = _listing(db), read_trace(db)
    if table not in listing["tables"] and table not in (trace or {}).get("tables", {}):
        raise HTTPException(404, f"Table inconnue : {table}")
    view = document_view(table, trace, load_semantics(db).get(table), load_fk_relations(db), listing["columns"].get(table))
    leaving = table not in listing["tables"]
    return {**view, "text": document_text(table, trace, db_cfg.dialect),
            "gone": leaving and listing["base"] is not None and table not in listing["base"],
            "unscoped": leaving and (listing["base"] is None or table in listing["base"])}


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
        raise HTTPException(400, f"Pas de questions d'évaluation pour {db} (offline/eval/datasets/{db.lower()}.yaml)")
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
        raise HTTPException(404, f"{db} n'a jamais été évaluée : lance l'évaluation depuis l'écran Index")
    with tempfile.TemporaryDirectory() as folder:
        html = build_dashboard([path], Path(folder) / "dashboard.html").read_text()
    return HTMLResponse(html)
