from pydantic import BaseModel, Field

class InsightCard(BaseModel):
    """Represents a single dashboard card with professional insights."""
    headline: str = Field(
        ..., 
        description="A concise, punchy title summarizing the performance aspect.",
        min_length=5,
        max_length=100
    )
    body: str = Field(
        ..., 
        description="Exactly 2-3 professional sentences comparing performance. Must be grounded in provided metrics.",
        min_length=50,
        max_length=500
    )
    hidden_story: str = Field(
        ..., 
        description="The technical 'why' or trade-off behind the data. Must reveal the underlying context.",
        min_length=20,
        max_length=300
    )

class EvaluationReport(BaseModel):
    """The final structured insights report for the UI dashboard."""
    recommendation_summary_text: str = Field(
        ..., 
        description=(
            "A 2-3 sentence executive summary declaring the winning strategy and explaining "
            "why it won based on the balance of accuracy and efficiency. Briefly mention "
            "any critical failures of the losing strategies."
            "keep the length of this summary in under 350 characters"
        ),
        min_length=50,
        max_length=350
    )
    retrieval_card: InsightCard = Field(..., description="Analysis of Recall, Precision, and MRR.")
    generation_card: InsightCard = Field(..., description="Analysis of Faithfulness and Answer Relevancy.")
    efficiency_card: InsightCard = Field(..., description="Analysis of Token Usage, Latency, and Ingestion.")