from langgraph.graph import MessagesState
from typing import TypedDict
from typing import Literal

class AgentState(MessagesState):
     question_complexity: Literal["simple", "complex", "out_of_scope"]
