# offline/studio/reading.py

"""
How the agent reads a table: its read mode in tables.yaml (lib/semantics._render_read_mode writes it in the
document, the READ MODES of the prompts tell what to do), and what the profile says of it: the partitions of
the table (Hive, Impala) and how often its key comes back across them. A human chooses the mode in the studio;
the studio proposes one from the data, never writes it (a wrong LAST silently drops rows).

  last    each partition is a full snapshot: the agent reads only the latest one (partition_column)
  all     each partition adds rows stored once: the agent reads them all, never deduplicates (partition_column)
  live    several versions of each key: the agent keeps the latest one (key_columns, version_column)
  (none)  the table read as it is

Proposed: last when the key of the table comes back in about every partition (one snapshot each), all when it
comes once over several partitions (increments). A partitioned table without a mode is a warning unless each of
its rows is stored once: the agent would count a row as many times as it is repeated.
"""

READ_MODES     = ("last", "all", "live")
NEEDS          = {"last": ("partition_column",), "all": ("partition_column",), "live": ("key_columns", "version_column")}
LAYOUT_KEYS    = ("read_mode", "key_columns", "partition_column", "version_column")
SNAPSHOT_SHARE = 0.8   # a snapshot per partition: the key back in at least this share of the partitions


def _number(n: float) -> str:
    return f"{n:,.1f}".replace(",", " ").replace(".", ",").removesuffix(",0")


def suggest(facts: dict) -> tuple[str | None, str]:
    """
    The mode the data of a partitioned table points to, and why; (None, why) when it says nothing. The key of
    the table is its identifier with the most values; its rows over its values tell in how many partitions a
    value comes back.
    """
    by, partitions = facts.get("partitioned_by") or [], facts.get("partitions") or 0
    keys = facts.get("keys") or {}
    if not by:
        return None, ""
    if len(by) > 1:
        return None, f"partitionnée par plusieurs colonnes ({', '.join(by)}) : un mode de lecture n'en suit qu'une"
    if partitions <= 1:
        return None, "une seule partition : photo complète ou ajout, rien ne permet encore de trancher"
    key = max(keys, key=lambda k: keys[k]["distinct"], default=None)
    if not key or not keys[key]["distinct"]:
        return None, "pas de clé mesurée : rien ne dit si les lignes se répètent d'une partition à l'autre"
    back = keys[key]["non_null"] / keys[key]["distinct"]
    if back >= SNAPSHOT_SHARE * partitions:
        return "last", (f"{key} revient dans {_number(back)} partitions sur {partitions} en moyenne : "
                        "chaque partition est une photo complète")
    if keys[key]["distinct"] == keys[key]["non_null"]:
        return "all", f"chaque {key} n'apparaît qu'une fois sur les {partitions} partitions : chacune ajoute des lignes"
    return None, (f"{key} revient {_number(back)} fois en moyenne sur {partitions} partitions : "
                  "ni une photo par partition, ni des ajouts nets")


def reading(entry: dict | None, facts: dict | None, columns: list[str] | None = None,
            partitioned_engine: bool = False) -> dict:
    """
    The read mode of a table as the studio shows it: the mode and its columns (tables.yaml), what is missing
    or unknown for the agent to build its document (`problems`: the agent fails), the partitions measured
    (None: not read, another engine or a profile older than their detection; `unread` on an engine that has
    them), the mode proposed and why, and `warn` when a partitioned table has no mode and its rows may repeat.
    """
    entry, facts = entry or {}, facts or {}
    mode = entry.get("read_mode")
    problems = []
    if mode is not None and mode not in READ_MODES:
        problems.append(f"mode inconnu : {mode}")
    for key in NEEDS.get(mode, ()):
        if not entry.get(key):
            problems.append({"partition_column": "il manque la colonne de partition", "key_columns": "il manque la clé",
                             "version_column": "il manque la colonne de version"}[key])
    used = [entry.get("partition_column"), *(entry.get("key_columns") or []), entry.get("version_column")]
    if columns:
        problems += [f"{c} n'est pas une colonne de la table" for c in dict.fromkeys(used) if c and c not in columns]
    suggested, why = suggest(facts)
    partitioned = bool(facts.get("partitioned_by"))
    return {
        "mode":             mode,
        "partition_column": entry.get("partition_column"),
        "key_columns":      list(entry.get("key_columns") or []),
        "version_column":   entry.get("version_column"),
        "problems":         problems,
        "partitions":       facts.get("partitions"),
        "partitioned_by":   list(facts.get("partitioned_by") or []),
        "unread":           partitioned_engine and bool(facts) and "partitions" not in facts,
        "suggested":        suggested,
        "why":              why,
        "warn":             partitioned and mode is None and suggested != "all",
    }


def edit_read_mode(entry: dict | None, mode: str | None, partition_column: str | None = None,
                   key_columns: list[str] | None = None, version_column: str | None = None,
                   columns: list[str] | None = None) -> dict:
    """
    The table block with its read mode chosen in the studio: last or all with their partition column, live with
    its key and version columns, None for the table read as it is (the four keys removed). The keys of another
    mode are removed. ValueError on an unknown mode, a missing column or one the table does not have (`columns`).
    """
    entry = {k: v for k, v in (entry or {}).items() if k not in LAYOUT_KEYS}
    if mode is None:
        return entry
    if mode not in READ_MODES:
        raise ValueError(f"Mode de lecture inconnu : {mode}")
    keys = list(dict.fromkeys(key_columns or []))
    chosen = {"partition_column": partition_column} if mode in ("last", "all") else {"key_columns": keys,
                                                                                       "version_column": version_column}
    if mode in ("last", "all") and not partition_column:
        raise ValueError("Choisis la colonne de partition")
    if mode == "live" and not keys:
        raise ValueError("Choisis la clé : la ou les colonnes qui désignent une même ligne d'une version à l'autre")
    if mode == "live" and not version_column:
        raise ValueError("Choisis la colonne de version")
    if mode == "live" and version_column in keys:
        raise ValueError(f"{version_column} ne peut pas être à la fois la clé et la version")
    named = [partition_column] if mode in ("last", "all") else [*keys, version_column]
    if columns and (unknown := [c for c in named if c not in columns]):
        raise ValueError(f"Colonne inconnue : {', '.join(unknown)}")
    return {**entry, "read_mode": mode, **chosen}
