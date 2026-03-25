from pydantic import BaseModel, Field


class QuestionScope(BaseModel):
    is_in_scope: bool = Field(
        description="True if the question can be answered from the database schema, False otherwise."
    )
    reason: str = Field(description="Brief explanation.")

# Structured output for node
class SelectedTables(BaseModel):
    tables: list[str] = Field(description="List of selected table names needed to answer the question.")
    reason: str = Field(description="Brief explanation of why these tables are needed and how they join.")
