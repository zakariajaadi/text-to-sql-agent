# src/schema_retrieval/prompts.py

"""
Prompts for the offline schema pipeline (schema_indexer.py, catalog_generators.py).
They run at indexing time, via make targets — never at query time.
Pipeline order: semantics → values → fk-relations → (review) → index → describe
"""

# ── Offline: catalog_generators.generate_semantics ──────────────────────
# Used by: catalog_generators.generate_semantics()
# Input context: DDL + sample rows for a single table, full table list for the database
# Expected output: TableSemantics — merged into config/semantics/<db>.yaml, then reviewed

TABLE_SEMANTICS_PROMPT = """
You are a data documentation expert describing a table of the {db_name} database.
Other tables in the database: {all_tables}

## TASK
Describe the table and each of its columns, for an analyst writing SQL queries.

## RULES
1. Table description: what one row represents, in one or two sentences.
2. Column description: one short sentence on its business meaning.
3. State units and currencies only when the column name or sample values make them
   explicit (e.g. amount_eur, duration_ms). Otherwise, if a unit matters, use "UNCLEAR:".
4. When sample values reveal codes or formats, explain them (e.g. "A = active, C = cancelled").
5. Do not describe joins: they are documented separately.
6. Never invent. If a column's meaning is unclear, start its description with "UNCLEAR:".
7. Describe every column of the DDL, spelled exactly as in the DDL.
8. Be concise: no filler like "The ..." and never paraphrase an explicit column name.
   If the name says it all (e.g. firstname, city), use a few words at most.
   Spend words only where the meaning is not obvious (ids pointing to other tables,
   codes, units, amounts).
9. Categorical: true when the column holds labels from a closed set a user would filter
   on: countries, states, cities, statuses, job titles, types, and the label of a reference
   table (aircraft model, fare class), even if each label appears once.
   False for identifiers, including ids pointing to another table (fare_class_id): the label
   lives in the referenced table. False also for names of people or companies, contact
   details (phone, fax, email, address), free text, dates, timestamps and amounts.

## EXAMPLE OUTPUT (style to imitate)
description: One row per customer order, with its shipping and billing details.
columns:
  order_id: Unique order identifier.                                  (categorical: false)
  firstname: First name.                                              (categorical: false)
  city: City.                                                         (categorical: true)
  cust_ref: Customer who placed the order.                            (categorical: false)
  st_cd: "Order status: A = active, C = cancelled, R = refunded."     (categorical: true)
  amt_ttc: UNCLEAR: amount including tax, currency not stated.        (categorical: false)

## TABLE
{ddl}
"""

# ── Offline: catalog_generators.infer_fk_relations ──────────────────────
# Used by: catalog_generators.infer_fk_relations()
# Input context: schemas of all tables in a database (DDL + reviewed semantics)
# Expected output: FKRelations — merged into config/fk_relations.yaml, then reviewed

FK_RELATIONS_PROMPT = """
You are a data modeling expert analyzing the {db_name} database.
The engine exposes no foreign key constraints: infer them from the schemas below.

## TASK
List every foreign key relationship: a column that references the identifier column
of another table (or of the same table, for hierarchies).

## RULES
1. Only use tables and columns that exist in the schemas below. Never invent names.
2. Write columns as table.column, spelled exactly as in the DDL.
3. The referenced column must be the identifier of its table (e.g. passenger.passenger_id).
4. Use column names, descriptions and sample values, including irregular names
   (e.g. booking.traveler_ref holding a passenger id under another name).
5. Only include relationships you are confident about. Never link columns that merely
   share a generic name (name, title, date).

## SCHEMAS
{schemas}
"""

# ── Offline: catalog_generators.describe_db ────────────────────────────────
# Used by: catalog_generators.describe_db()
# Input context: schemas of all tables in a database (DDL + reviewed semantics)
# Expected output: comma-separated keywords, written to config/db_descriptions.yaml (shown in the chat profile selector)

DB_DESCRIPTION_PROMPT = """
You are a database documentation expert.

Given the enriched schemas of all tables in a database, write a concise description
of the database as a whole.

## OUTPUT FORMAT
A single comma-separated list of keywords describing:
- The business domain (e.g. air travel, logistics)
- The main entities (e.g. flights, airports, passengers)
- The main processes (e.g. booking, check-in)

## RULES
- Max 15 keywords
- No full sentences — keywords only
- Output ONLY the comma-separated list, nothing else

## DATABASE: {db_name}

## SCHEMAS
{schemas}
"""