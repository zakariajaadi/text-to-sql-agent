# offline/cli.py

"""
CLI of the semantic layer pipeline: generate the semantic layer, review it, then index it.

    python -m offline.cli [command] [--db NAME]     (PYTHONPATH=src, see the Makefile)

    profile       → semantic_layer/<db>/profile.yaml        facts measured on the data (no LLM), and the
                    formats and empty columns it detects into tables.yaml (a table that has none yet)
                    (--tables a,b / --missing: only those tables, merged into the profile)
    semantics     → semantic_layer/<db>/tables.yaml         descriptions, categorical columns
                    (--fresh: generated again from scratch, the manual keys kept)
    fk-relations  → semantic_layer/<db>/relationships.yaml  FK relations (Hive / Impala)
    metrics       → semantic_layer/<db>/metrics.yaml        business metrics
    (review the YAML files)
    index         → Chroma schema index + value index data/values/<db>.sqlite (default command)

Without --db, the command runs on every configured database.
"""

import argparse
import sys

from loguru import logger

from config import cfg
from offline import query_cancel
from offline.indexer.schema_indexer import SchemaIndexer
from offline.semantic_gen import metrics, profiling, relationships, tables
from resources import db_registry


def main():
    """CLI entry point: one command, on one database (--db) or all of them."""
    parser = argparse.ArgumentParser(prog="offline.cli")
    parser.add_argument("command", nargs="?", default="index",
                        choices=["index", "profile", "semantics", "fk-relations", "metrics"])
    parser.add_argument("--db", help="Database name (default: all databases)")
    parser.add_argument("--fresh", action="store_true",
                        help="semantics: regenerate the descriptions and categorical columns, "
                             "keeping the manual keys (formats, exclude_columns, data_as_of...)")
    parser.add_argument("--tables", type=lambda s: [t.strip() for t in s.split(",") if t.strip()],
                        help="profile: only these tables (comma-separated), the others kept in profile.yaml")
    parser.add_argument("--missing", action="store_true",
                        help="profile: only the tables not in profile.yaml yet")
    args = parser.parse_args()

    commands = {
        "index":        lambda name: SchemaIndexer(name).run(),
        "profile":      lambda name: profiling.profile_database(
                            db_registry[name], name, tables=args.tables, missing=args.missing,
                            personal=cfg.get_db(name).exclude_personal_data),
        "semantics":    lambda name: tables.generate_table_semantics(name, fresh=args.fresh),
        "fk-relations": relationships.infer_fk_relations,
        "metrics":      metrics.generate_metrics,
    }

    # A stop (the studio, Ctrl+C) cancels the running query on the cluster, not just this process
    query_cancel.install(db._engine for db in db_registry.values())

    targets = [cfg.get_db(args.db).name] if args.db else [db.name for db in cfg.databases]
    try:
        for name in targets:
            commands[args.command](name)
    except KeyboardInterrupt:
        logger.warning(f"{args.command} stopped")
        sys.exit(130)


if __name__ == "__main__":
    main()
