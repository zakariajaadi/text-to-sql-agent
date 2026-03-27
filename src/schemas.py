from pydantic import BaseModel, Field


class RankedTablesAndScope(BaseModel):
    tables: list[str] = Field(description="List of selected table names needed to answer the question.")
    is_in_scope: bool = Field(description="True if the question can be answered from the selected tables, False otherwise.")
    reason: str = Field(description="Brief explanation of table selection and scope decision.")
 
