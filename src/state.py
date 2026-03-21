from typing import Literal

from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     question_complexity: Literal["simple", "complex", "out_of_scope"]
     query_plan: str  # populated only for complex questions
     remaining_steps: RemainingSteps
