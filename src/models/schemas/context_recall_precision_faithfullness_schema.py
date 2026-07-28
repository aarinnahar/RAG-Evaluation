from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# 1. Sub-Models for Individual Analysis Rows
# =====================================================================

class RecallFactAnalysis(BaseModel):
    atomic_fact: str = Field(
        description="A single, independent proposition extracted directly from the Golden Answer."
    )
    found_in_chunks: bool = Field(
        description="True if this specific fact is explicitly stated or unambiguously inferred within the retrieved chunks. Otherwise, False."
    )
    verbatim_source_quote: Optional[str] = Field(
        default=None,
        description="The exact substring quote from the retrieved chunks that verifies this fact. Must be null if found_in_chunks is False."
    )
    reasoning: str = Field(
        description="A brief, objective explanation justifying whether the chunks support or omit this specific fact."
    )


class PrecisionChunkAnalysis(BaseModel):
    chunk_index: int = Field(
        description="The zero-indexed position (0, 1, 2...) of the chunk being evaluated from the original retrieved chunks list."
    )
    is_relevant_to_question: bool = Field(
        description="True if this specific chunk contains information that directly helps answer the user's question. False if it is noise or filler text."
    )
    reasoning: str = Field(
        description="A brief, objective explanation of why this specific chunk is relevant or irrelevant to the question."
    )


class FaithfulnessStatementAnalysis(BaseModel):
    generated_statement: str = Field(
        description="A single, distinct claim or factual assertion extracted from the Generated Answer."
    )
    supported_by_chunks: bool = Field(
        description="True only if this statement is 100% grounded in and supported by the retrieved chunks. False if it contains any hallucination or unverified assumption."
    )
    reasoning: str = Field(
        description="A clear breakdown showing how the chunks support this statement, or pointing out the specific hallucinated element."
    )


# =====================================================================
# 2. Main Parent Model for the Unified Evaluator Node
# =====================================================================

class RagEvaluationAudit(BaseModel):
    """
    The complete structural payload containing raw semantic signals 
    for calculating Context Recall, Context Precision, and Faithfulness 
    deterministically in code.
    """
    context_recall_analysis: List[RecallFactAnalysis] = Field(
        description="A granular list breaking down the Golden Answer to check if its core components were successfully retrieved."
    )
    context_precision_analysis: List[PrecisionChunkAnalysis] = Field(
        description="An ordered evaluation of the retrieved chunks to score the alignment and ranking precision of the retriever."
    )
    faithfullness_analysis: List[FaithfulnessStatementAnalysis] = Field(
        description="A granular analysis of the generated output to check for text grounding and pinpoint hallucinations."
    )