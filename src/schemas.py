from typing import Literal

from pydantic import BaseModel, Field


# Structured output for node
class QuestionComplexity(BaseModel):
    question_complexity: Literal["simple", "complex", "out_of_scope"] = Field(
        description=(
            "'out_of_scope' if the question cannot be answered from the schema. "
            "'simple' if it requires 1-2 tables and basic filtering. "
            "'complex' if it requires 3+ tables, advanced aggregations, or subqueries."
        )
    )
    reason: str = Field(description="Explain why this category was chosen.")

# Structured output for node
class SelectedTables(BaseModel):
    tables: list[str] = Field(description="List of selected table names needed to answer the question.")
    reason: str = Field(description="Brief explanation of why these tables are needed and how they join.")
