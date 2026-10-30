from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     standalone_question: str            # self-contained rewrite of user question (output of condense_question)
     selected_db: str                    # target database name (output of route_database)
     is_in_scope: bool                   # whether the question can be answered (output of rerank_tables; set to False by sql_coder and sql_validator when the information is in no column)
     unavailable_reason: str             # why a question gets no query: "restricted" (withheld, confidential), "absent" (in no data), empty otherwise (rerank_tables, sql_coder, sql_validator)
     candidate_tables: list[str]         # tables of the database the selector reads, the closest to the question first: all of them up to retrieval.catalog_max (output of retrieve_schema)
     selected_schemas: dict[str, str]    # table_name → full DDL of the tables kept by rerank_tables, FK-expanded, from the schema store (output of rerank_tables)
     retrieved_schema: str               # DDL sent to planner and sql_coder, columns pruned (output of prune_columns)
     llm_selected_tables: list[str]      # tables selected by reranker LLM, before FK expansion, used by eval for precision/recall
     selected_metrics: list[str]         # business metrics the question asks for, from semantic_layer/<db>/metrics.yaml (output of rerank_tables)
     clarify_criteria: list[str]         # metrics that could define the question's criterionless superlative, offered as clarification options; their tables are not loaded (output of rerank_tables, clarify only)
     filter_values: list[str]            # values the question filters on, as written and as likely spelled in the database (output of rerank_tables)
     matched_values: dict[str, dict[str, list[str]]]   # table → column → values of the value index close to filter_values (output of match_values)
     values_not_found: bool              # filter_values searched, none close to any value of the data (output of match_values)
     clarify: bool                       # the planner may ask the user one question (set by the caller: app.py from the db config; the eval never sets it)
     cancel_key: str                     # key of the turn's cancel event (set by app.py): the user's Stop cancels the running query on the cluster (lib/query_cancel.py)
     clarification_question: str         # question asked to the user instead of a plan (output of ask_clarification)
     clarification_options: list[str]    # its options, shown as clickable buttons by the Chainlit UI (output of ask_clarification)
     query_plan: str                     # execution plan from planner node
     generated_query: str                # SQL query (output of sql_coder)
     validation_feedback: str            # error message, empty on success (output of sql_validator)
     validation_error: str               # bare error of the failed query, shown by the Chainlit UI (output of sql_validator)
     validation_check: str               # check the query failed: "guardrail", "scope" (reads tables out of the scope), "excluded" (reads excluded columns), "values", "database" or "unreachable" (no database to run it), empty on success (output of sql_validator)
     result_row_count: int               # rows read from the query result, at most max_rows (output of sql_validator)
     result_truncated: bool              # query returned more than max_rows rows (output of sql_validator)
     result_columns: list[str]           # column names of the query result (output of sql_validator)
     result_rows: list[list]             # JSON-safe result rows, rendered by the Chainlit result card (output of sql_validator)
     remaining_steps: RemainingSteps     # retry counter for sql_coder loop (managed by langgraph)

