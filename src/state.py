from typing import Literal

from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     question_complexity: Literal["simple", "complex", "out_of_scope"]
     remaining_steps: RemainingSteps
