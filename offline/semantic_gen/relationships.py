# offline/semantic_gen/relationships.py

"""
Offline generator of the FK relations: semantic_layer/<db>/relationships.yaml,
for engines that expose no FK metadata (Hive, Impala). The relations measured on the data
come first (profile.yaml, no LLM), the LLM adds the ones named differently.

Every relation carries its evidence, read by offline/studio (the agent reads from and to only):
  found  : share of the distinct values of `from` found in `to`, on a sample (key_overlap)
  origin : profile (a same-name key whose values match), llm, or manual (added in the studio)
A relation of the LLM is measured too and kept whatever its share: under KEY_OVERLAP the studio
shows it in amber, for a human to keep or ignore.

Additive: existing and ignored relations are never modified, but for the evidence they lack
(check_relations, `fk-relations --check`). It reads the database directly (DDL + reviewed
semantics), never Chroma, so it can run before the first index.

Pipeline order: profile → semantics → relations → metrics → (review) → index   (see offline/cli.py)
"""

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from lib.semantics import load_profile
from offline.semantic_gen.profiling import key_overlap, key_relations
from offline.semantic_gen.prompts import FK_RELATIONS_PROMPT
from offline.semantic_gen.schemas import FKRelations
from offline.semantic_gen.utils import (
    load_relations_file,
    render_schemas,
    table_columns,
    write_relations_file,
)
from resources import db_registry, model

# A same-name key is a relation when this share of its values is found in the hub table
KEY_OVERLAP = 0.9

# ------------------------------------------------------------
# Chain
# ------------------------------------------------------------

# All schemas → FK relations between tables
fk_relations_chain = (
    ChatPromptTemplate.from_messages([("human", FK_RELATIONS_PROMPT)])
    | model.with_structured_output(FKRelations)
)

# ------------------------------------------------------------
# Generator
# ------------------------------------------------------------

def infer_fk_relations(db_name: str):
    """
    Infers Foreign Key relations and merges them into semantic_layer/<db>/relationships.yaml:
    first the keys unique in one table and present under the same name in others (make profile),
    kept when KEY_OVERLAP of their values are found in that table; then the LLM's.

    Meant for engines that expose no FK metadata (Hive, Impala).
    Relations pointing to unknown tables or columns are dropped.

    Additive only: existing and ignored relations are never modified.
    """
    db = db_registry[db_name]

    # 1. Relations measured on the data, no LLM: a key unique in one table and present under the
    #    same name in others (make profile), kept when the values match (same name ≠ same entity)
    valid = []
    for source, target in key_relations(load_profile(db_name)):
        overlap = key_overlap(db, source, target)
        if overlap is None:
            logger.info(f"[{db_name}] Not checked: {source} -> {target} (no value), left to the LLM")
        elif overlap >= KEY_OVERLAP:
            valid.append({"from": source, "to": target, "found": round(overlap, 3), "origin": "profile"})
        else:
            logger.info(f"[{db_name}] Not a relation: {source} -> {target} ({overlap:.0%} of the values found)")

    # 2. Infer the other relations from the schemas (DDL + reviewed semantics)
    logger.info(f"[{db_name}] Asking the LLM for the other relations")
    inferred = fk_relations_chain.invoke({
        "db_name": db_name,
        "schemas": render_schemas(db, db_name),
    }).relations

    # 3. Drop relations referencing unknown tables or columns (LLM hallucinations)
    columns = table_columns(db)

    def exists(ref: str) -> bool:
        table, _, column = ref.partition(".")
        return column in columns.get(table, [])

    for r in inferred:
        if exists(r.from_column) and exists(r.to_column):
            if (r.from_column, r.to_column) not in {(v["from"], v["to"]) for v in valid}:
                valid.append({"from": r.from_column, "to": r.to_column, "origin": "llm"})
        else:
            logger.warning(f"[{db_name}] Rejected unknown relation: {r.from_column} -> {r.to_column}")

    # 4. Merge — only add relations neither declared nor ignored
    entry = load_relations_file(db_name)
    known = {(r["from"], r["to"]) for r in entry["relations"] + entry["ignore"]}
    added = [r for r in valid if (r["from"], r["to"]) not in known]
    entry["relations"].extend(added)

    # 5. The evidence of every relation that has none: the LLM's, and the ones written before
    unchecked = [relation for relation in entry["relations"] if "found" not in relation]
    logger.info(f"[{db_name}] {len(unchecked)} relation(s) to check")
    for relation in unchecked:
        measure(db, db_name, relation)
        if relation.get("origin") == "llm" and (relation["found"] or 0) < KEY_OVERLAP:
            logger.warning(f"[{db_name}] To check: {relation['from']} -> {relation['to']} "
                           f"({relation['found'] or 0:.0%} of the values found), kept for review")

    write_relations_file(db_name, entry)
    logger.info(f"[{db_name}] {len(added)} new relation(s) added to relationships.yaml — review them.")


def check_relations(db_name: str):
    """Measure again the evidence (found) of every declared relation, no LLM (`fk-relations --check`)."""
    db    = db_registry[db_name]
    entry = load_relations_file(db_name)
    logger.info(f"[{db_name}] {len(entry['relations'])} relation(s) to check")
    for relation in entry["relations"]:
        measure(db, db_name, relation)
    write_relations_file(db_name, entry)
    logger.info(f"[{db_name}] {len(entry['relations'])} relation(s) checked on the data")


def measure(db, db_name: str, relation: dict) -> None:
    """Set `found` on a relation: the share of its values found (None when `from` holds no value)."""
    logger.info(f"[{db_name}] Checking: {relation['from']} -> {relation['to']}")
    try:
        overlap = key_overlap(db, relation["from"], relation["to"])
    except Exception as e:
        logger.warning(f"[{db_name}] Not checked: {relation['from']} -> {relation['to']} — {str(e).splitlines()[0][:200]}")
        return
    relation["found"] = None if overlap is None else round(overlap, 3)
