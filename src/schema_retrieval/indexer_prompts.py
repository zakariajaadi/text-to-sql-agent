# src/schema_retrieval/indexer_prompts.py

"""
Prompts for the offline schema pipeline (schema_indexer.py, generators.py).
They run at indexing time, via make targets — never at query time.
Pipeline order: semantics → fk-relations → (review) → index → describe
"""

# ── Offline: generators.generate_semantics ──────────────────────────────
# Used by: generators.generate_semantics()
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

## EXAMPLE OUTPUT (style to imitate)
description: One row per customer order, with its shipping and billing details.
columns:
  order_id: Unique order identifier.
  firstname: First name.
  city: City.
  cust_ref: Customer who placed the order.
  st_cd: "Order status: A = active, C = cancelled, R = refunded."
  amt_ttc: UNCLEAR: amount including tax, currency not stated.

## TABLE
{ddl}
"""

# ── Offline: generators.infer_fk_relations ──────────────────────────────
# Used by: generators.infer_fk_relations()
# Input context: enriched schemas of all tables in a database (from Chroma)
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
3. The referenced column must be the identifier of its table (e.g. customer.customerid).
4. Use column names, descriptions and sample values, including irregular names
   (e.g. a column holding an employee id under another name).
5. Only include relationships you are confident about. Never link columns that merely
   share a generic name (name, title, date).

## SCHEMAS
{schemas}
"""

# ── Offline: schema_indexer.extract (fallback) ───────────────────────────────
# Used by: schema_indexer.enrich_chain, called by SchemaIndexer.extract()
#          only for tables without an entry in config/semantics/<db>.yaml
# Input context: DDL + sample rows for a single table, full table list for the database
# Expected output: original DDL prepended with a business description comment block

SCHEMA_DESCRIPTION_PROMPT = """
You are a database documentation expert.

Given the DDL and sample rows of a database table, write a concise business description.

## TASK 
Add a comment block at the TOP of the DDL, following this exact structure:
-- Description: <what this table represents in business terms>
-- Granularity: <what one row represents, e.g. "one line item within an order", "one customer">
-- Business role: <what business process it supports>
-- Common joins: <tables from the database context it's typically joined with and why>
-- Key columns: <most important columns and what they represent>

## RULES
- Be concise — max 2 lines per field
- Infer from column names and sample data, do not invent
- Only reference tables that exist in the list below for Common joins
- Output ONLY the comment block + the original DDL, nothing else

## DATABASE CONTEXT
Database: {db_name}
All tables in this database: {all_tables}

## DDL
{ddl}
"""

# ── Offline: generators.describe_db ────────────────────────────────────────
# Used by: generators.describe_db()
# Input context: indexed schemas of all tables in a database (from Chroma)
# Expected output: comma-separated keywords, written to config/db_descriptions.yaml (shown in the chat profile selector)

DB_DESCRIPTION_PROMPT = """
You are a database documentation expert.

Given the enriched schemas of all tables in a database, write a concise description
of the database as a whole.

## OUTPUT FORMAT
A single comma-separated list of keywords describing:
- The business domain (e.g. music store, e-commerce)
- The main entities (e.g. artists, albums, customers)
- The main processes (e.g. invoicing, order management)

## RULES
- Max 15 keywords
- No full sentences — keywords only
- Output ONLY the comma-separated list, nothing else

## DATABASE: {db_name}

## SCHEMAS
{schemas}
"""