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


CONDENSE_QUESTION_PROMPT = """\
Given the conversation history and the following question, can you rephrase the user's \
question in its original language so that it is self-sufficient. You are presented \
with a conversation that may contain some spelling mistakes and grammatical errors, \
but your goal is to understand the underlying question. Make sure to avoid the use of \
unclear pronouns.

If the question is already self-sufficient, return the original question. If it seem \
the user is authorizing the chatbot to answer without specific context, make sure to \
reflect that in the rephrased question.

Chat history: {chat_history}

Question: {question}
""" 

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