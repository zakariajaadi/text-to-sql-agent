# offline/eval/history.py
# ===============
# History of the evals of a database: one line per saved state of its results, so the studio draws the score
# over time (the results file keeps only the last run).
#
# Usage (CLI, from the project root, PYTHONPATH=src):
#   python -m offline.eval.history --db Chinook --backfill     # seed it from the git history of results/<db>.json
#
# Written next to the results, results/<db>.history.jsonl, each time run.py saves them (a whole run, a partial
# rerun, a rescore). A line: {"at", "kind", "commit", "questions", "scored", "exact"}; counts only, never a query.

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from loguru import logger

from offline.eval.dashboard import RESULTS_DIR

RUN, PARTIAL, RESCORE, GIT = "run", "partial", "rescore", "git"


def history_path(results: str | Path) -> Path:
    """results/<db>.json → results/<db>.history.jsonl"""
    path = Path(results)
    return path.with_name(f"{path.stem}.history.jsonl")


def score(results: list[dict]) -> dict:
    """The questions of a run, how many were compared on their rows (exec_exact set), how many returned the expected rows."""
    scores = [r.get("exec_exact") for r in results]
    scored = [s for s in scores if s is not None]
    return {"questions": len(scores), "scored": len(scored), "exact": sum(1 for s in scored if s)}


def record(results_file: str | Path, results: list[dict], kind: str, at: str | None = None, commit: str | None = None) -> dict:
    """Append the state of results just saved to the history (at: ISO minutes, now by default)."""
    line = {"at": at or datetime.now().isoformat(timespec="minutes"), "kind": kind, "commit": commit, **score(results)}
    path = history_path(results_file)
    with path.open("a") as f:
        f.write(json.dumps(line) + "\n")
    return line


def read_history(results_file: str | Path) -> list[dict]:
    """The lines of the history, oldest first; [] without a history. A line that can't be read is skipped."""
    path = history_path(results_file)
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().splitlines():
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return sorted(lines, key=lambda line: line["at"])


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=RESULTS_DIR).stdout


def backfill(results_file: str | Path) -> int:
    """
    Seed the history from the git history of the results file (renames followed): one line per committed version
    whose results hold scores and differ from the version before (a file moved is the same run), dated by its run
    (meta.run_at) or else its commit. A version already in the history (same date) is skipped. Returns the number
    of lines added.
    """
    path = Path(results_file).resolve()
    known = {line["at"] for line in read_history(path)}
    added, previous = [], None
    log = _git("log", "--follow", "--format=@%h %cI", "--name-only", "--", str(path))   # paths from the top, newest first
    for entry in reversed(log.split("@")[1:]):   # --reverse would stop at a rename
        lines = [x for x in entry.splitlines() if x]
        if len(lines) < 2:
            continue
        commit, committed = lines[0].split()
        try:
            data = json.loads(_git("show", f"{commit}:{lines[1]}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        results = data.get("results") if isinstance(data, dict) else data
        if not results or not any("exec_exact" in r for r in results) or results == previous:
            continue
        previous = results
        meta = (data.get("meta") or {}) if isinstance(data, dict) else {}
        at = meta.get("run_at") or datetime.fromisoformat(committed).astimezone().replace(tzinfo=None).isoformat(timespec="minutes")
        if at in known:
            continue
        known.add(at)
        added.append({"at": at, "kind": GIT, "commit": meta.get("commit") or commit, **score(results)})
    if added:
        merged = sorted([*read_history(path), *added], key=lambda line: line["at"])
        history_path(path).write_text("".join(json.dumps(line) + "\n" for line in merged))
    return len(added)


def main():
    parser = argparse.ArgumentParser(description="History of the evals of a database")
    parser.add_argument("--db", required=True)
    parser.add_argument("--backfill", action="store_true", help="Seed the history from the git history of the results")
    args = parser.parse_args()
    results = RESULTS_DIR / f"{args.db.lower()}.json"
    if args.backfill:
        logger.info(f"{backfill(results)} run(s) added to {history_path(results)}")
    for line in read_history(results):
        print(f"{line['at']}  {line['exact']}/{line['scored']}  ({line['kind']})")


if __name__ == "__main__":
    main()
