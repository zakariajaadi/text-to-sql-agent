from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     selected_db: str
     is_in_scope: bool                  
     candidate_schemas: dict[str, str]  # table_name → DDL (output of node retrieve_schema)
     retrieved_schema: str              # Filtred DDL (output of node rerank_tables)
     generated_query: str               # SQL produced by sql_coder
     validation_feedback: str           # error feedback from sql_validator (empty on success)
     remaining_steps: RemainingSteps    # managed counter to handle infinite loop at parent level
     standalone_question: str           # rewritten question after condensation

