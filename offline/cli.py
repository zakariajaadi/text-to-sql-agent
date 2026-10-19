# offline/cli.py

"""
CLI of the semantic layer pipeline: generate the semantic layer, review it, then index it.

    python -m offline.cli [command] [--db NAME]     (PYTHONPATH=src, see the Makefile)

    semantics     → semantic_layer/<db>/tables.yaml         descriptions, categorical columns
                    (--fresh: generated again from scratch, the manual keys kept)
    fk-relations  → semantic_layer/<db>/relationships.yaml  FK relations (Hive / Impala)
    metrics       → semantic_layer/<db>/metrics.yaml        business metrics
    (review the YAML files)
    index         → Chroma schema index + value index data/values/<db>.sqlite (default command)

Without --db, the command runs on every configured database.
"""

import argparse

from config import cfg
from offline.indexer.schema_indexer import SchemaIndexer
from offline.semantic_gen import metrics, relationships, tables


def main():
    """CLI entry point: one command, on one database (--db) or all of them."""
    parser = argparse.ArgumentParser(prog="offline.cli")
    parser.add_argument("command", nargs="?", default="index",
                        choices=["index", "semantics", "fk-relations", "metrics"])
    parser.add_argument("--db", help="Database name (default: all databases)")
    parser.add_argument("--fresh", action="store_true",
                        help="semantics: regenerate the descriptions and categorical columns, "
                             "keeping the manual keys (formats, exclude_columns, data_as_of...)")
    args = parser.parse_args()

    commands = {
        "index":        lambda name: SchemaIndexer(name).run(),
        "semantics":    lambda name: tables.generate_table_semantics(name, fresh=args.fresh),
        "fk-relations": relationships.infer_fk_relations,
        "metrics":      metrics.generate_metrics,
    }

    targets = [cfg.get_db(args.db).name] if args.db else [db.name for db in cfg.databases]
    for name in targets:
        commands[args.command](name)


if __name__ == "__main__":
    main()
