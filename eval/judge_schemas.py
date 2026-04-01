# eval/judge_schemas.py
# ===========================
# Pydantic schemas and constants for the LLM-as-judge evaluation.

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Single dimension score returned by the LLM judge."""
    score: int = Field(ge=1, le=5, description="Score from 1 (wrong) to 5 (perfect)")
    reasoning: str = Field(description="Brief justification for the score")


class JudgeEvaluation(BaseModel):
    """
    Complete evaluation returned by the LLM judge.
    Used with .with_structured_output() to avoid manual JSON parsing.
    """
    table_selection: DimensionScore = Field(description="Are the correct tables used?")
    column_selection: DimensionScore = Field(description="Are the correct columns selected?")
    join_correctness: DimensionScore = Field(description="Are JOINs correct and complete?")
    filter_logic: DimensionScore = Field(description="Are WHERE/HAVING conditions accurate?")
    aggregation: DimensionScore = Field(description="Are GROUP BY, COUNT, SUM, etc. correct?")
    overall: DimensionScore = Field(description="Overall query correctness considering all dimensions")

# Dimension names, derived from the Pydantic model
# avoids hardcoding dimension names elsewhere
DIMENSIONS = list(JudgeEvaluation.model_fields.keys())

# Human-readable labels for score values, used in print_summary output
SCORE_LABELS = {5: "perfect", 4: "good", 3: "partial", 2: "poor", 1: "wrong"}