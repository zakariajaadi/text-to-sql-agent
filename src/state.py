from typing import Literal

from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     question_complexity: Literal["simple", "complex", "out_of_scope"]
     query_plan: str  # populated only for complex questions
     candidate_schemas: dict[str, str]  # table_name → DDL (output of node retrieve_schema)
     retrieved_schema: str              # Filtred DDL (output of node rerank_tables)
     remaining_steps: RemainingSteps
