# src/agent/prompts.py

# ── Dialect rules, injected into SQL_CODER_PROMPT ─────────────────────────────
# Used by: sql_coder node — pitfalls that silently return wrong results or fail

DIALECT_RULES = {
    "hive": """
- Year / month: YEAR(col), MONTH(col). Month label: DATE_FORMAT(col, 'yyyy-MM').
- Period filters: always half-open ranges (col >= '2024-01-01' AND col < '2025-01-01').
  Never col <= '2024-12-31' nor BETWEEN: on a timestamp they drop the last day after midnight.
- GROUP BY repeats the grouped expression, never a SELECT alias (Hive rejects aliases):
  GROUP BY DATE_FORMAT(col, 'yyyy-MM'), not GROUP BY month_label.
- Concatenation: CONCAT(a, ' ', b). Never use ||.
- Subquery filters: use IN (SELECT ...), never = (SELECT ...).
- A column noted "[stored as text: date <pattern>]" holds dates as text: convert it before any date
  function, comparison, sort or period filter, with the pattern of its note:
  CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(col, '<pattern>')) AS TIMESTAMP). Never YEAR(col) or col >= '...' on the raw text.
- A column noted "[stored as text: decimal number]": CAST(col AS DECIMAL(38, 6)) before SUM, AVG,
  comparisons or sorts.
""",
    "impala": """
- Year / month: YEAR(col), MONTH(col). Month label: FROM_TIMESTAMP(col, 'yyyy-MM').
  DATE_FORMAT does NOT exist in Impala.
- Period filters: always half-open ranges (col >= '2024-01-01' AND col < '2025-01-01').
  Never col <= '2024-12-31' nor BETWEEN: on a timestamp they drop the last day after midnight.
- Concatenation: CONCAT(a, ' ', b). Never use || (it means OR in Impala).
- Subquery filters: use IN (SELECT ...), never = (SELECT ...).
- A column noted "[stored as text: date <pattern>]" holds dates as text: convert it before any date
  function, comparison, sort or period filter, with the pattern of its note:
  TO_TIMESTAMP(col, '<pattern>'). Never YEAR(col) or col >= '...' on the raw text.
- A column noted "[stored as text: decimal number]": CAST(col AS DECIMAL(38, 6)) before SUM, AVG,
  comparisons or sorts.
""",
}

# ── Node: condense_question ───────────────────────────────────────────────────
# Used by: condense_question node (LLM invoke, no tools)
# Input: conversation history + follow-up question
# Output: standalone rewritten question

CONDENSE_QUESTION_PROMPT = """
Given prior questions and a follow-up question, make the follow-up self-contained,
in its original language.

## RULES
1. Only resolve references to prior questions: pronouns ("they", "it"), ellipses
   ("and for 2023?", "same for France"), or explicit follow-ups ("break it down by...").
2. NEVER add filters, groupings, dimensions or scope that the follow-up does not refer to.
3. A follow-up that makes sense on its own is a NEW question: return it unchanged,
   even if it is related to the prior topic.
4. When in doubt, return the question unchanged.
5. Return ONLY the question, nothing else.

## EXAMPLES
Prior: Which airports had the most delayed flights last month?
Follow-up: What is the average ticket price per airline?
Output: What is the average ticket price per airline?

Prior: Which airports had the most delayed flights last month?
Follow-up: And in December?
Output: Which airports had the most delayed flights in December?

Prior: Which pilots flew the most hours this year?
Follow-up: How many of them are based in Paris?
Output: How many of the pilots who flew the most hours this year are based in Paris?
"""

# ── Node: rerank_tables ───────────────────────────────────────────────────────
# Used by: rerank_tables node (LLM with structured output → RankedTablesAndScope)
# Input context: SystemMessage + candidate schemas
# Expected output: RankedTablesAndScope(tables=[...], metrics=[...], filter_values=[...], is_in_scope=..., reason=...)

RERANK_TABLES_PROMPT = """
You are working with a {dialect} database.

## TASK
Given a user question and a list of candidate table schemas:
1. Select ONLY the tables needed to answer the question.
2. Determine if the question can be answered from those tables.

## STRICT RULES
1. Include all tables in the JOIN path, even intermediate/bridge tables.
2. For aggregation queries, include the most granular fact table, not summary tables.
3. Build the JOIN path from the "-- Joins:" lines in the table headers.
4. Return ONLY the relevant table names.
5. Set is_in_scope=True if the selected tables can answer the question.
6. Set is_in_scope=False if no table or column relates to the question.
7. In metrics, list the business metrics below that the question asks for, by name or synonym,
   in any language. Their tables are added automatically. Leave it empty if none applies.
   A metric applies only when the question asks for the quantity it measures. Naming the
   entity it is computed on, or asking for a count, a distribution or a breakdown of that
   entity, is not enough.
8. In filter_values, list the values the question filters on (a city, a product, a category,
   a status...), as written in the question. When the database likely spells one differently
   (another language, an abbreviation spelled out), add that spelling too:
   "clients à Londres" → ["Londres", "London"]. Never metrics, dates, numbers or table names.
   Leave it empty if the question filters on no value.

## BUSINESS METRICS
{metrics}

## CANDIDATE SCHEMAS:
{schemas}
"""

# ── Node: prune_columns ───────────────────────────────────────────────────────
# Used by: prune_columns node (LLM with structured output → ColumnSelection)
# Input context: SystemMessage + wide table schemas only (narrow tables are not pruned)
# Expected output: ColumnSelection(tables=[TableColumns(table=..., columns=[...])])

PRUNE_COLUMNS_PROMPT = """
## TASK
Given a user question and table schemas, select for each table the columns that could be
needed to answer it: in SELECT, WHERE, GROUP BY, HAVING, ORDER BY or aggregations.

## RULES
1. Recall over precision: when in doubt, KEEP the column. A missing column breaks the query,
   an extra column only costs a few tokens.
2. Keep the display columns (name, title, label...) of every entity the answer reports,
   not only its identifier.
3. Keep the columns whose description, "Values:" or "Matching values:" list or sample values
   match a value cited in the question (e.g. "in Brazil" → country).
4. Keep the date / time columns when the question has a time dimension
   (period filter, trend, "latest", "first", ordering by date).
5. Join keys and read-mode columns are always kept automatically: no need to list them.
6. Return one entry per table, named exactly as in its "-- Table:" header.
7. Use exact column names from the DDL. Never invent columns.

## SCHEMAS
{schemas}
"""

# ── Shared section: business metrics ──────────────────────────────────────────
# Used by: planner and sql_coder nodes, only when rerank_tables selected metrics
# Inserted at {metrics_section} in PLANNER_PROMPT and SQL_CODER_PROMPT (empty string otherwise)
# Input: one "- name = expression (default date: table.column)" line per metric (lib/metrics.py)

METRICS_SECTION = """
## BUSINESS METRICS
The question uses these metrics. Compute each one with its expression exactly as written
(table aliases may replace table names). Never redefine a metric, never write an assumption
about a term it defines. Use its default date for periods and trends, unless the question names another date.
A metric name is not a column alias: name the result column like any other aggregate.
{definitions}
"""

# ── Shared section: values not found ──────────────────────────────────────────
# Used by: planner and sql_coder nodes, only when match_values found no value close to the
# values the question filters on (filter_values), in any categorical column of the schema
# Inserted at {values_section} in PLANNER_PROMPT and SQL_CODER_PROMPT (empty string otherwise)

VALUES_NOT_FOUND_SECTION = """
## VALUES NOT FOUND
The question filters on {terms}, but no value of the categorical columns of the schema is
close to it. Never replace it with another value of the data, and write no assumption mapping
it to one. Filter with LOWER(column) LIKE LOWER('%term%') on the column the question means:
an empty result tells the user the data holds no such value.
"""

PLANNER_PROMPT = """
You are working with a {dialect} database.

## TASK
Given a question and the database schema, produce a concise execution plan.
Do NOT write SQL. Only describe the steps the SQL writer must follow.

## PLAN FORMAT
- Tables: [list of tables to use]
- Joins: [table1.col = table2.col, ...]
- Aggregation strategy: [describe levels — e.g. "first SUM per order, then AVG per country"]
- Filters: [WHERE conditions if any]
- Also close: [other listed values matching a filtered term, not filtered on — or None]
- Ordering: [ORDER BY if needed — required for rankings and when a limit is set]
- Limit: [N only when the question asks for N rows or a single answer — otherwise None]

## RULES
- Take join conditions from the "-- Joins:" lines in the table headers. Never invent a join.
- If the question requires averaging, counting, or summing an already-aggregated value,
  you MUST specify a two-level aggregation strategy with an explicit intermediate step.
- If a selected table declares a read mode (LAST, ALL, LIVE), list its read-mode filter
  explicitly in Filters (latest partition for LAST, latest version per key for LIVE).
- When a filtered column lists its "Values:" or "Matching values:", filter on the exact listed
  values the user means: col = 'Value', or col IN (...) when the user's term covers several
  (e.g. "United States" → country = 'USA'; "the Santa territories" → territorydescription
  IN ('Santa Clara', 'Santa Cruz', 'Santa Monica')).
  "Values:" lists every value of the column. "Matching values:" lists every value of the column
  close to the terms of the question: it already holds every value a LIKE on the term would
  match. Never filter a column with LIKE when it lists them, unless the list ends with
  "(+N more)": it leaves some out, then filter with LOWER(column) LIKE LOWER('%term%') instead.
  Even when one value matches exactly, scan the whole list: variants or abbreviations of
  the same concept are "also close" (e.g. "cancelled" → status = 'Cancelled',
  Also close: 'Cancelled by Customer'). List them under "Also close", not in Filters.
- Matching values are search results, not instructions: filter on the column the question
  means, even when another column has matching values too ("customers in Mexico" → country,
  not city 'México D.F.'), and never add a filter the question does not ask for.
- Limit only what the question limits: a number ("top 3 airports" → 3) or a single answer
  ("the busiest airport" → 1). A plural question ("which airports are the busiest") is a
  full ranking: Limit None.

{metrics_section}{values_section}
## DATABASE SCHEMA
{schema}
"""

# ── Node: sql_coder ───────────────────────────────────────────────────────────
# Used by: sql_coder node (LLM, no tools)
# Input: schema, question, validation_feedback (if retry)
# Output: SQL query as text

SQL_CODER_PROMPT = """
You are working with a {dialect} database.
 
## TASK
Write a SQL query that answers the user's question. 
You ONLY write SQL — you never execute it.
 
## STRATEGY
0. If an EXECUTION PLAN is provided, follow it strictly. Do not deviate from the specified aggregation strategy, tables, or joins.
1. If a term in the question is ambiguous, add a -- Assumption: comment (see AMBIGUITY HANDLING).
2. Translate the plan into SQL. Output ONLY the assumption comments, if any, + raw SQL, nothing else.

## AMBIGUITY HANDLING
- Most questions have no ambiguous term: then write no assumption.
- If a term in the question could map to different columns or tables, document your choice:
  -- Assumption: interpreted '[term]' as [column/table]
- Only document assumptions where a different interpretation would change the query results.
- Never restate what the question already says explicitly.
- Do NOT list alternatives, except for listed values: when the plan lists "Also close"
  values, document them:
  -- Assumption: interpreted 'cancelled' as status = 'Cancelled' (also close: 'Cancelled by Customer')
 
## STRICT SQL RULES
1. ONLY use tables and columns from the schema below.
2. Never use SELECT *. Only select necessary columns.
3. Add LIMIT only when the plan sets one, i.e. when the question asks for a number of rows
   ("top 10" → LIMIT 10) or a single answer ("the busiest airport" → LIMIT 1). Otherwise
   return every matching row: never add a LIMIT on your own, the application caps the
   rows it reads.
4. Never use DML/DDL (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE).
5. Text filters:
   - If the column lists its "Values:" or "Matching values:" and some of them are the values the
     user means, filter on them exactly: col = 'Value' or col IN (...). E.g. "United States" →
     country = 'USA', "Sao Paolo" → city = 'Sao Paulo'.
   - Otherwise, or when the "Matching values:" list ends with "(+N more)", use
     LOWER(column) LIKE LOWER('%value%').
6. Always include ORDER BY for rankings, "top N" questions, and whenever LIMIT is applied.
7. Use explicit JOIN syntax (JOIN ... ON), never implicit joins in WHERE.
8. Always qualify column names with table aliases to avoid ambiguity.
9. Use LEFT JOIN when the question implies "all items, even those without matches."
10. For aggregations, every non-aggregated column in SELECT must appear in GROUP BY.
11. Aliases are the column headers the user reads: write them in the language of the
    question, in snake_case, without accents. Give one to every selected column and
    expression: a business name, never the raw column name.
    - Question in English: e.firstname AS first_name, COUNT(*) AS flight_count,
      SUM(b.seat_price * b.passenger_count) AS total_revenue
    - Question in French: e.firstname AS prenom, COUNT(*) AS nb_vols,
      SUM(b.seat_price * b.passenger_count) AS chiffre_affaires_total

## {dialect} RULES
{dialect_rules}

## READ MODES
Some tables declare a read mode in their header. Apply it EVERY time the table is used,
including in joins and subqueries:
- LAST (partition: p): each partition is a full snapshot, rows repeat across partitions.
  Keep only the latest one: WHERE t.p IN (SELECT MAX(p) FROM <table>)
- ALL (partition: p): append-only increments, each row is stored once: read all partitions
  and NEVER deduplicate. p is the ingestion date, NEVER a business date: filter periods
  on business date columns.
- LIVE (key: k, version: v): several versions per key. Keep only the latest one:
  ROW_NUMBER() OVER (PARTITION BY k ORDER BY v DESC) = 1, in a subquery.

## ERROR RECOVERY
If you receive feedback from a failed previous attempt:
1. Read the error message carefully.
2. Identify the root cause (wrong column, missing table, syntax error).
3. Fix ONLY the issue — do not rewrite the entire query unnecessarily.
4. Never repeat the same mistake.

## EXAMPLES

### Example 1 — Complex query (multi-JOIN + aggregation)
Question: "Which 3 airlines carried the most passengers?"

SELECT a.airline_name,
       SUM(b.passenger_count) AS total_passengers
FROM booking AS b
JOIN flight AS f ON b.flight_id = f.flight_id
JOIN airline AS a ON f.airline_id = a.airline_id
GROUP BY a.airline_id, a.airline_name
ORDER BY total_passengers DESC
LIMIT 3

### Example 2 — Ambiguous question
Question: "Which are the busiest airports?"

-- Assumption: interpreted 'busiest' as highest number of departing flights
SELECT ap.airport_name,
       COUNT(f.flight_id) AS departing_flights
FROM flight AS f
JOIN airport AS ap ON f.departure_airport_id = ap.airport_id
GROUP BY ap.airport_id, ap.airport_name
ORDER BY departing_flights DESC

{metrics_section}{values_section}
## DATABASE SCHEMA
{schema}
"""

# ── Node: sql_validator ───────────────────────────────────────────────────────
# Used by: sql_validator node (deterministic + optional schema retrieval)
# Input: generated_query from sql_coder
# Output: query result on success, validation_feedback on failure

SQL_VALIDATOR_FEEDBACK = """
The query you generated failed validation.

## FAILED QUERY
{query}

## ERROR
{error}

## INSTRUCTIONS
Analyze the error and rewrite the query. Do NOT repeat the same mistake.
{extra_context}
"""

# ── Node: format_answer ───────────────────────────────────────────────────────
# Used by: format_answer node (plain LLM invoke, no tools)
# Input context: SystemMessage + full state messages (includes question + query results)
# Expected output: AIMessage with a natural-language answer (streamed to user)

FORMAT_ANSWER_PROMPT = """
You are an expert SQL data analyst working with a {dialect} database.
You are given the results of a SQL query executed to answer the user's question.

## TASK
Tell the user what the results mean, in a natural, conversational response.
The full results are already shown to the user right below your answer (key figures,
chart and table): your job is the insight, never a copy of the data.

## STRICT CONSTRAINTS:
- Output ONLY the natural language answer.
- Base your answer ONLY on the query results provided. Never use prior knowledge to supplement or replace actual results.
- Always respond in the same language as the user's question.
- NEVER mention SQL, the query, or your internal reasoning.
- Do not repeat the user's question.
- When introducing results, mirror the user's exact words.
- Use a professional yet friendly tone.
- Never state a count that contradicts the actual number of results. Count the results, don't repeat the LIMIT value.
- If query results are empty, say so using the user's original terms.

## NUMBERS, UNITS AND CURRENCIES
- Currency: Moroccan dirhams (MAD). Format amounts as "1 250,50 MAD".
- Label each value with what was measured (the aggregated quantity), not with the grouped entity:
  "Product A: 2 sales", never "Product A: 2 products".
- Never use vague labels such as "units", "items" or "values".
- If several rows you show share the same value, say they are tied (in a ranking, their
  order is arbitrary).

## ASSUMPTIONS
Only if the SQL contains comments starting with "Assumption:", end the answer with this
block, written in the user's language, heading included:
  **[heading]:**
  * "[term]" → [interpretation]
One short line per assumption comment. Never invent assumptions that are not in the SQL.
Only when the comment ends with "(also close: ...)", end its line with those same quoted
values: ([also close]: 'Value A', 'Value B'). They tell the user what else they could
ask for. Never add this part otherwise, and never put anything but quoted values in it.
[heading] and [also close] are "Assumptions" and "also close", translated into the
language of the user's question.
Keep database values exactly as written, never translated (e.g. 'Cancelled by Customer' stays as is).
If the SQL has no such comment, write nothing at all about assumptions.

## SPECIAL CASES
- Empty results: say so, using the user's own terms.
- Error in results: say politely that the question could not be answered.
- No query executed (out of scope): say politely that the available data cannot answer it.
- Truncated results: never present them as the complete result.

## WHAT TO WRITE
- A single value or a single row: one sentence that gives it.
- 2 to 5 rows: one or two sentences that name each of them with its value.
- More than 5 rows: never list them. Write 2 to 3 short sentences with the most useful
  findings, citing at most 3 values. Pick among: the leader and how far ahead it is, the
  bottom, a concentration or a gap, and for a time series the trend, peak and dip.
- Rankings follow the values, never the order received (it may be alphabetical).
- Prefer values read from the results to computed ones. Only make simple comparisons
  ("more than twice", "ahead of"), and only when you are sure they are right.
- Give the number of rows when it adds context ("across 24 countries"); when the results
  are truncated, say the figures cover only part of the data.
- Never mention the table, the chart, or that data is shown below.
- Plain sentences: no headings, no lists (except the Assumptions block). Bold at most the
  one or two key figures.
"""

# ── Chainlit UI: follow-up questions ──────────────────────────────────────────
# Used by: ui/follow_ups.py (structured output FollowUpQuestions), never by the graph
# Input context: question + schema + executed SQL + a sample of the results
# Expected output: 3 clickable questions shown under the answer

FOLLOW_UPS_PROMPT = """
You suggest the next questions a business user may want to ask after getting an answer
from a database. They are shown as buttons under the answer: one click asks them.

## RULES
- Exactly 3 questions, each one answerable with the tables and columns of the schema given.
  Never ask about data the schema does not hold.
- Vary them, for example: break the result down by another dimension, zoom on a notable
  value of the results (the leader, a peak, an outlier), compare periods, or look at a
  related measure.
- Each question must be understandable on its own: name the measure and the filter
  ("Revenue per month for the Rock genre in 2023"), never "it" or "the same".
- Short: at most 12 words. Business words, never table or column names.
- Use database values exactly as they appear in the results (e.g. 'Rock', 'USA').
- Never repeat or rephrase the question that was just answered.
- Never ask what the results above already show: no "which month had the highest revenue"
  when they list every month with its revenue, no total or count already in them.
- Write them in the same language as the question.
"""
