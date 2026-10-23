# offline/eval/dashboard.py
# =================
# Eval dashboard — one self-contained HTML page built from the results JSON files.
#
# Usage (CLI, from the project root — see `make eval-dashboard`):
#   python -m offline.eval.dashboard                            # every results/<db>.json → results/dashboard.html
#   python -m offline.eval.dashboard --db Chinook Northwind     # only these databases (keeps private results out)
#   python -m offline.eval.dashboard --output /tmp/report.html
#
# The page needs no server: data, styles and scripts are inlined. Each question is joined with its
# dataset entry (accepted queries, expected tables, notes) so the page can show the gold answer.

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
DATASETS_DIR = EVAL_DIR / "datasets"
TEMPLATE = EVAL_DIR / "eval_lib" / "dashboard_template.html"
DATA_PLACEHOLDER = '"__EVAL_DATA__"'

# French label of each dataset `category` (the SQL skill a question tests), shown by the page.
# A category missing here is shown as its code, and the build warns about it.
CATEGORY_LABELS = {
    # Plain SQL
    "count": "Comptage",
    "filtering": "Filtrage",
    "aggregation": "Agrégation",
    "group_by": "Regroupement",
    "having": "Filtre sur un regroupement (HAVING)",
    "ranking": "Classement (top N)",
    "nested_aggregation": "Agrégation à deux niveaux",
    "union": "Union de plusieurs tables",
    "intersection": "Intersection de deux populations",
    "anti_join": "Recherche d'absence (anti-jointure)",
    # Joins
    "join_aggregation": "Jointure et agrégation",
    "multi_join_aggregation": "Jointures multiples et agrégation",
    "multi_join_filter": "Jointures multiples et filtre",
    "bridge_join": "Jointure par une table de correspondance",
    "join_text_key": "Jointure entre clé texte et clé numérique",
    "island_join": "Jointure vers un groupe de tables isolé",
    "table_choice": "Choix entre des tables proches",
    # Dates and amounts
    "date": "Calcul sur des dates",
    "date_filter": "Filtre sur une période",
    "date_aggregation": "Agrégation par période",
    "date_text": "Date stockée en texte",
    "date_int": "Date stockée en nombre",
    "data_as_of": "Période relative à la date des données",
    "amount_text": "Montant stocké en texte",
    "dedup_amount": "Montant répété sur plusieurs lignes",
    # Business codes and scope
    "code_filter": "Filtre sur un code métier",
    "ambiguous": "Question ambiguë",
    "out_of_scope": "Question hors périmètre",
}


def _db_name(stem: str) -> str:
    """The database name as config.yaml spells it (results files are lower-case: northwind.json → Northwind)."""
    try:
        from config import cfg
        return next((db.name for db in cfg.databases if db.name.lower() == stem.lower()), stem)
    except Exception:  # no config at hand (e.g. run outside the project): the file name will do
        return stem


def _git_commit(path: Path) -> str | None:
    """Last commit that touched the results file, for runs saved before results carried their own meta."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(path)],
            capture_output=True, text=True, cwd=EVAL_DIR, check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_dataset(db: str) -> dict[str, dict]:
    """Dataset entries by question (missing file → no gold answers shown, the page still works)."""
    path = DATASETS_DIR / f"{db.lower()}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return {q["question"]: q for q in yaml.safe_load(f) or []}


def load_run(results_file: Path) -> dict:
    """
    One database's run, as the page reads it.
    1. Read results + summary (+ meta when the run saved it)
    2. Fill missing meta from the file (date) and git (commit)
    3. Attach the dataset's accepted queries, expected tables and notes to each result
    """
    # 1. Results
    data = json.loads(results_file.read_text())
    db = data.get("meta", {}).get("db") or _db_name(results_file.stem)

    # 2. Meta
    meta = {
        "db": db,
        "run_at": datetime.fromtimestamp(results_file.stat().st_mtime).isoformat(timespec="minutes"),
        "commit": _git_commit(results_file),
        **data.get("meta", {}),
    }

    # 3. Gold answers
    dataset = _load_dataset(db)
    results = []
    for r in data["results"]:
        q = dataset.get(r["question"], {})
        results.append({
            **r,
            "accepted_queries": q.get("accepted_queries", []),
            "expected_tables": q.get("expected_tables", []),
            "notes": (q.get("notes") or "").strip() or None,
            # compare: judge set in the dataset after the run counts too (rows not compared)
            **({"compare": "judge"} if q.get("compare") == "judge" or r.get("compare") == "judge" else {}),
        })

    return {"meta": meta, "summary": data["summary"], "results": results}


def build_dashboard(results_files: list[Path], output: Path) -> Path:
    """Inline the runs into the template and write the page."""
    runs = [load_run(p) for p in results_files]
    unlabeled = sorted({r["category"] for run in runs for r in run["results"]} - CATEGORY_LABELS.keys())
    if unlabeled:
        logger.warning(f"No French label for {', '.join(unlabeled)}: shown as is (add them to CATEGORY_LABELS)")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "categories": CATEGORY_LABELS,
        "runs": runs,
    }
    # "</" would close the inline <script> early
    data_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    body = TEMPLATE.read_text().replace(DATA_PLACEHOLDER, data_js)
    html = f'<!doctype html>\n<html lang="fr">\n{body}</html>\n'

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    logger.info(f"Dashboard saved to {output} ({len(payload['runs'])} databases)")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build the eval dashboard (HTML)")
    parser.add_argument("--db", nargs="+", help="Databases to include (default: every results/<db>.json)")
    parser.add_argument("--output", default=str(RESULTS_DIR / "dashboard.html"))
    args = parser.parse_args()

    if args.db:
        files = [RESULTS_DIR / f"{db.lower()}.json" for db in args.db]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            parser.error(f"No results for: {', '.join(missing)} (run `make eval DB=...` first)")
    else:
        files = sorted(RESULTS_DIR.glob("*.json"))
        if not files:
            parser.error(f"No results in {RESULTS_DIR} (run `make eval DB=...` first)")

    build_dashboard(files, Path(args.output))


if __name__ == "__main__":
    main()
