# offline/eval/show.py
# ============
# Print the eval results of some questions as plain text, to read or paste an eval miss:
# the agent's SQL next to the accepted queries, the tables, the judge's notes. No result rows
# (the results file holds none), so nothing of the data itself.
#
# Usage (CLI, from the project root — see `make eval-show`):
#   python -m offline.eval.show --db awb_silver                 # every miss: rows differ, or refused when it should answer
#   python -m offline.eval.show --db awb_silver --only agence   # the questions holding this word (repeatable)

import argparse
import json

from offline.eval.dashboard import CATEGORY_LABELS
from offline.eval.eval_lib import load_questions
from offline.eval.run import dataset_path, results_path


def is_miss(r: dict) -> bool:
    """Rows different from the gold, or an in-scope question left without a query (technical errors aside)."""
    if r.get("error"):
        return False
    if r["category"] == "out_of_scope":
        return r["judge"]["overall"]["score"] < 5
    return r.get("exec_exact") == 0 or not r.get("generated_sql")


def block(r: dict, q: dict) -> str:
    """One question as text: header, tables, the agent's SQL, the accepted queries, the judge."""
    exec_text = {1: "lignes identiques", 0: "lignes différentes"}.get(r.get("exec_exact"), "non comparé")
    lines = [
        "=" * 100,
        r["question"],
        f"{r['difficulty']} · {CATEGORY_LABELS.get(r['category'], r['category'])} · juge {r['judge']['overall']['score']}/5 · {exec_text}",
        f"Tables attendues : {', '.join(q.get('expected_tables', [])) or '–'}",
        f"Tables choisies  : {', '.join(r.get('selected_tables', [])) or '–'}",
    ]
    if q.get("notes"):
        lines.append(f"Note du jeu de test : {q['notes'].strip()}")
    lines += ["", "--- SQL de l'agent", r.get("generated_sql") or "(aucune requête)"]
    for i, gold in enumerate(q.get("accepted_queries", []), 1):
        lines += ["", f"--- Référence {i}", gold.strip()]
    lines += ["", "--- Juge LLM"]
    lines += [f"{dim} {v['score']}/5 : {v['reasoning']}" for dim, v in r["judge"].items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Print eval results of some questions (default: the misses)")
    parser.add_argument("--db", default="Chinook")
    parser.add_argument("--only", action="append", metavar="WORD", help="Questions holding this word (repeatable)")
    args = parser.parse_args()

    results = json.loads(results_path(args.db).read_text())["results"]
    gold = {q["question"]: q for q in load_questions(dataset_path(args.db))}

    if args.only:
        words = [w.lower() for w in args.only]
        chosen = [r for r in results if any(w in r["question"].lower() for w in words)]
    else:
        chosen = [r for r in results if is_miss(r)]

    for r in chosen:
        print(block(r, gold.get(r["question"], {})))
    print("=" * 100)
    print(f"{len(chosen)} question(s) sur {len(results)}")


if __name__ == "__main__":
    main()
