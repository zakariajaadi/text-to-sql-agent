from pydantic import BaseModel, Field


# Output of the rerank_tables node (nodes.py)
class RankedTablesAndScope(BaseModel):
    tables: list[str] = Field(description="List of selected table names needed to answer the question.")
    metrics: list[str] = Field(default_factory=list, description="Names of the business metrics the question asks for, exactly as listed. Empty if none.")
    is_in_scope: bool = Field(description="True if the question can be answered from the selected tables, False otherwise.")
    reason: str = Field(description="Brief explanation of table selection and scope decision.")

# One table of the prune_columns output (nodes.py)
class TableColumns(BaseModel):
    table:   str       = Field(description="Table name, exactly as in the '-- Table:' header")
    columns: list[str] = Field(description="Columns of this table needed to answer the question, spelled exactly as in the DDL")


# Output of the prune_columns node (nodes.py): semantic columns only,
# structural columns (join keys, read mode) are added by lib/column_pruning.py
class ColumnSelection(BaseModel):
    tables: list[TableColumns] = Field(description="Selected columns of every table in the schemas, one entry per table")


# Follow-up questions suggested under an answer (ui/follow_ups.py, Chainlit UI only)
class FollowUpQuestions(BaseModel):
    questions: list[str] = Field(description="3 short follow-up questions the user may ask next, answerable from the schema")
