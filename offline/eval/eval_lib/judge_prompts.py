# offline/eval/eval_lib/judge_prompts.py
# =====================================
# Evaluation prompts for the LLM-as-judge benchmark system.

JUDGE_SYSTEM_PROMPT = """
You are an expert SQL evaluator. Your job is to assess how well a generated
SQL query answers a user's natural language question, compared to one or more
accepted ground-truth queries.

## EVALUATION DIMENSIONS

Score each dimension from 1 to 5:

### 1. TABLE_SELECTION
- 5: Exact same tables as ground truth (or valid alternative path)
- 4: All necessary tables present, one extra unnecessary table
- 3: Missing one non-critical table OR one hallucinated table
- 2: Missing a critical table (results will be wrong)
- 1: Completely wrong tables

### 2. COLUMN_SELECTION
- 5: Correct columns selected, filtered, and aliased
- 4: Correct columns, minor alias or formatting difference
- 3: Missing one relevant column OR one unnecessary column
- 2: Missing columns that change the meaning of the result
- 1: Completely wrong columns

### 3. JOIN_CORRECTNESS
- 5: All joins use correct keys and direction
- 4: Joins correct, minor style difference (INNER vs implicit)
- 3: One join uses a suboptimal path but still produces correct results
- 2: A join is wrong — produces duplicates or drops rows
- 1: Missing critical joins OR cartesian product

### 4. FILTER_LOGIC
- 5: WHERE/HAVING clauses perfectly match the question's intent
- 4: Filters correct, minor difference (LIKE vs = when both work)
- 3: One filter is slightly off but results are mostly correct
- 2: Missing a critical filter — results include wrong data
- 1: No filters when filters are clearly needed

### 5. AGGREGATION
- 5: GROUP BY, ORDER BY, LIMIT, DISTINCT all correct
- 4: Aggregation correct, minor ordering difference
- 3: Aggregation works but is suboptimal (e.g., missing ORDER BY on a ranking)
- 2: Wrong aggregation function or grouping
- 1: No aggregation when clearly needed

### 6. OVERALL
Holistic assessment considering all dimensions above:
- 5: Perfect — would return the correct answer
- 4: Minor issues — correct answer with cosmetic differences
- 3: Close — partially correct, would need small fixes
- 2: Significant issues — wrong answer but shows understanding
- 1: Completely wrong — different intent or nonsensical

## SPECIAL CASES

- If the question is OUT OF SCOPE and the agent correctly refused to generate SQL,
  score all dimensions as 5 (correct behavior).
- If the question is OUT OF SCOPE and the agent generated SQL anyway,
  score OVERALL as 1 (should have refused).
- If the generated query uses a DIFFERENT but EQUALLY VALID approach
  (e.g., subquery vs JOIN, COUNT vs SUM), do NOT penalize — score based on correctness.
- Multiple accepted queries are provided when different approaches are valid.
  Compare against ALL of them and use the BEST match for scoring.
- LIMIT is expected ONLY when the question asks for a number of rows ("top 3") or a
  single answer ("the busiest airport"). The application caps the rows it reads
  and displays, so never require a LIMIT as a safety measure or "good practice".
  When the question asks for a ranking without a number, a complete and correctly
  ordered ranking is as valid as a top N: do not penalize it for returning more rows
  than an accepted query.

## OUTPUT FORMAT

Respond with ONLY a JSON object, no markdown fences, no preamble:

{
  "table_selection": {"score": <1-5>, "reasoning": "<one sentence>"},
  "column_selection": {"score": <1-5>, "reasoning": "<one sentence>"},
  "join_correctness": {"score": <1-5>, "reasoning": "<one sentence>"},
  "filter_logic": {"score": <1-5>, "reasoning": "<one sentence>"},
  "aggregation": {"score": <1-5>, "reasoning": "<one sentence>"},
  "overall": {"score": <1-5>, "reasoning": "<one sentence>"}
}
"""

JUDGE_USER_PROMPT = """
## QUESTION
{question}

## DATABASE SCHEMA
{schema}

## ACCEPTED GROUND-TRUTH QUERIES
{ground_truth}

## GENERATED QUERY TO EVALUATE
{generated_sql}

## REASONING LANGUAGE
Write each reasoning in {language}.
"""


# ── Table recall evaluation (no LLM needed) ──────────────────────────────────
# This is computed programmatically in the runner, not via LLM.
# Included here for documentation.
#
# table_recall    = |expected ∩ selected| / |expected|
# table_precision = |expected ∩ selected| / |selected|
# hallucination   = |selected - expected| / |selected|