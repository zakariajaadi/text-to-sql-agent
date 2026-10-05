# src/agent/prompts.py

# ── Dialect rules, injected into SQL_CODER_PROMPT ─────────────────────────────
# Used by: sql_coder node — pitfalls that silently return wrong results or fail

DIALECT_RULES = {
    "hive": """
- Year / month: YEAR(col), MONTH(col). Month label: DATE_FORMAT(col, 'yyyy-MM').
- Period filters: prefer ranges (col >= '2024-01-01' AND col < '2025-01-01').
- Concatenation: CONCAT(a, ' ', b). Never use ||.
- Subquery filters: use IN (SELECT ...), never = (SELECT ...).
""",
    "impala": """
- Year / month: YEAR(col), MONTH(col). Month label: FROM_TIMESTAMP(col, 'yyyy-MM').
  DATE_FORMAT does NOT exist in Impala.
- Period filters: prefer ranges (col >= '2024-01-01' AND col < '2025-01-01').
- Concatenation: CONCAT(a, ' ', b). Never use || (it means OR in Impala).
- Subquery filters: use IN (SELECT ...), never = (SELECT ...).
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
# Expected output: RankedTablesAndScope(tables=[...], is_in_scope=..., reason=...)

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
3. Keep the columns whose description or sample values match a value cited in the question
   (e.g. "in Brazil" → country).
4. Keep the date / time columns when the question has a time dimension
   (period filter, trend, "latest", "first", ordering by date).
5. Join keys and read-mode columns are always kept automatically: no need to list them.
6. Return one entry per table, named exactly as in its "-- Table:" header.
7. Use exact column names from the DDL. Never invent columns.

## SCHEMAS
{schemas}
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
- Ordering: [ORDER BY if needed — required when results will be limited]

## RULES
- Take join conditions from the "-- Joins:" lines in the table headers. Never invent a join.
- If the question requires averaging, counting, or summing an already-aggregated value,
  you MUST specify a two-level aggregation strategy with an explicit intermediate step.
- If a selected table declares a read mode (LAST, ALL, LIVE), list its read-mode filter
  explicitly in Filters (latest partition for LAST, latest version per key for LIVE).

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
1. Identify any ambiguous terms in the question. For each, add a -- Assumption: comment.
2. Translate the plan into SQL. Output ONLY the assumption comments + raw SQL, nothing else.

## AMBIGUITY HANDLING
- If a term in the question could map to different columns or tables, document your choice:
  -- Assumption: interpreted '[term]' as [column/table]
- Only document assumptions where a different interpretation would change the query results.
- Never restate what the question already says explicitly.
- Do NOT list alternatives.
 
## STRICT SQL RULES
1. ONLY use tables and columns from the schema below.
2. Never use SELECT *. Only select necessary columns.
3. Always LIMIT results to {limit} rows. Only exceptions:
   - Single aggregate value (COUNT, AVG, SUM): no LIMIT needed.
   - Questions asking for a complete list: use LIMIT 100 as a safety cap.
   - User specifies an explicit number (e.g. "top 10"): use that number.
4. Never use DML/DDL (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE).
5. For text matching, use LOWER(column) LIKE LOWER('%value%').
6. Always include ORDER BY for rankings, "top N" questions, and whenever LIMIT is applied.
7. Use explicit JOIN syntax (JOIN ... ON), never implicit joins in WHERE.
8. Always qualify column names with table aliases to avoid ambiguity.
9. Use LEFT JOIN when the question implies "all items, even those without matches."
10. For aggregations, every non-aggregated column in SELECT must appear in GROUP BY.
11. Always give aggregates and computed expressions a descriptive snake_case alias
    (e.g. COUNT(*) AS customer_count, SUM(il.unitprice * il.quantity) AS total_revenue).

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
LIMIT 5
 
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
Your job is to transform these results into a natural, conversational response.

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
- If several results share the same value at the end of a ranking, say they are tied.

## ASSUMPTIONS
Only if the SQL contains comments starting with "Assumption:", end the answer with:
  **Assumptions:**
  * Assumed "[term]" → [interpretation]
One short line each. Never invent assumptions that are not in the SQL.
If the SQL has no such comment, write nothing at all about assumptions.

## SPECIAL CASES
- Empty results: say so, using the user's own terms.
- Error in results: say politely that the question could not be answered.
- No query executed (out of scope): say politely that the available data cannot answer it.

## FORMATTING
- Several results: a clean Markdown list.
- Keep it concise.
"""
