"""
condense_question.py

Reformulates a follow-up question into a standalone question
using the conversation history.

Input:  {"question": str, "chat_history": list}
Output: str (standalone question)
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from prompts import CONDENSE_QUESTION_PROMPT



def condense_question_chain(llm: BaseChatModel) -> Runnable:
    """
    Chain that rewrites a question as a standalone question.

    Ensures the retriever always gets a self-sufficient query,
    even for follow-up questions like "And how do we fix it?".

    Input:  {"question": str, "chat_history": list}
    Output: str
    """
    prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
    return prompt | llm | StrOutputParser()