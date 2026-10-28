# offline/semantic_gen/prompts.py

"""
Prompts of the semantic layer generators (tables.py, relationships.py, metrics.py).
They run offline, via make targets — never at query time.
Pipeline order: profile → semantics → relations → metrics → (review) → index
"""

# ── Offline: tables.generate_table_semantics ──────────────────────────────
# Used by: tables.generate_table_semantics()
# Input context: DDL + sample rows for a single table, full table list for the database
# Expected output: TableSemantics — merged into semantic_layer/<db>/tables.yaml, then reviewed

TABLE_SEMANTICS_PROMPT = """
You are a data documentation expert describing a table of the {db_name} database.
Other tables in the database: {all_tables}

## TASK
Describe the table and each of its columns, for an analyst writing SQL queries.

## RULES
1. Table description: what one row represents, in one or two sentences, consistent with the
   ROW FACTS measured on the data: when an identifier has fewer distinct values than the
   table has rows, say the table holds several rows per that entity (e.g. "several rows per
   order"). Never write "one row per X" unless the facts show one row per X.
2. Column description: one short sentence on its business meaning.
3. State units and currencies only when the column name or sample values make them
   explicit (e.g. amount_eur, duration_ms). Otherwise, if a unit matters, say it is not stated.
4. When sample values reveal codes or formats, explain them (e.g. "A = active, C = cancelled").
5. Do not describe joins: they are documented separately.
6. Never invent. Unclear: true when the column's meaning, or a unit that matters, cannot be told
   from its name and sample values; then describe what you can and say what is missing.
   False otherwise.
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
10. Write the descriptions in {language}, the language users ask their questions in. Keep the
   column names and the codes exactly as they are.

## EXAMPLE OUTPUT (style to imitate)
description: One row per customer order, with its shipping and billing details.
columns:
  order_id: Unique order identifier.                                  (categorical: false)
  firstname: First name.                                              (categorical: false)
  city: City.                                                         (categorical: true)
  cust_ref: Customer who placed the order.                            (categorical: false)
  st_cd: "Order status: A = active, C = cancelled, R = refunded."     (categorical: true)
  amt_ttc: Amount including tax, currency not stated.                 (unclear: true, categorical: false)

## ROW FACTS
{facts}

## TABLE
{ddl}
"""

# ── Offline: relationships.infer_fk_relations ─────────────────────────────
# Used by: relationships.infer_fk_relations()
# Input context: schemas of all tables in a database (DDL + reviewed semantics)
# Expected output: FKRelations — merged into semantic_layer/<db>/relationships.yaml, then reviewed

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

# ── Offline: metrics.generate_metrics ─────────────────────────────────────
# Used by: metrics.generate_metrics()
# Input context: tables holding measures and the tables they join to (DDL + reviewed semantics),
#                column roles found deterministically, metrics already defined or rejected
# Expected output: MetricProposals — validated, merged into semantic_layer/<db>/metrics.yaml, then reviewed

METRICS_PROMPT = """
You are a business intelligence expert defining the key metrics of the {db_name} database ({dialect}).
Analysts will ask questions about these metrics; an agent will compute them with your expressions as is.

## TASK
Propose the business metrics analysts would most often ask about: amounts, volumes, counts,
averages and ratios. At most 8, the most useful first. Fewer is fine.

## RULES
1. expr is ONE aggregate SQL expression (SUM, COUNT, AVG, MIN, MAX, or a ratio of aggregates),
   never a full query. Write every column as table.column, spelled exactly as in the schemas.
2. Build amounts from the most detailed fact table (e.g. order lines, not order headers), so the
   metric can be broken down by product, category or any related entity.
3. Apply the adjustments the schema documents (discounts, refunds, signs...) and say in the
   description what the metric includes and excludes (e.g. "discounts applied, freight excluded").
4. Averages per entity divide by the distinct count of that entity: "per order, not per order line"
   → SUM(...) / COUNT(DISTINCT order_lines.order_id). Never AVG over line-level rows for them.
5. Count entities with COUNT(DISTINCT table.id), never COUNT(*). DISTINCT is only for counting
   entities: never SUM(DISTINCT ...) or AVG(DISTINCT ...) on an amount, as two rows with the same
   amount are two amounts. Never propose a metric that only counts one column (COUNT(DISTINCT
   table.id) alone): the agent counts entities by itself. A count is a metric only with a business
   rule inside (a CASE on a status, a ratio), e.g. COUNT(DISTINCT CASE WHEN ... THEN table.id END).
6. time_dimension: the business date of the event the metric measures (order date, invoice date),
   from the fact table or from a table it joins to through the "-- Joins:" lines. Empty if none.
7. Only use measures, identifiers and dates listed in COLUMN ROLES. Never invent a column.
8. Names in snake_case, unique, built from the words of this database's tables and columns
   (count of invoice rows → invoice_count, not order_count), without prefixes such as
   total_ or sales_ (revenue, not total_revenue or sales_revenue).
   Never propose a metric listed in EXISTING METRICS or REJECTED METRICS, nor another name
   for one of their expressions.
9. Synonyms: 3 to 8 words users may use for the metric, in {languages}.
10. Write the descriptions in {language}, the language users ask their questions in.

## COLUMN ROLES
{roles}

## EXISTING METRICS
{existing}

## REJECTED METRICS
{rejected}

## SCHEMAS
{schemas}
"""
