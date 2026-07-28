LLM_INSIGHTS_GENERATION_PROMPT = """You are an elite Data Scientist analyzing RAG (Retrieval-Augmented Generation) evaluation metrics.
Your task is to populate the requested reporting schema with punchy, executive-level insights based on the provided metrics.

### CRITICAL INSTRUCTIONS
1. NO MATH: Do not perform any mathematical calculations. Rely strictly on the exact numbers and percentages provided in the payload.
2. NO EXTRAPOLATION: Your ONLY source of truth is the Input Data below. Do not invent, infer, or add trends. If the data does not support a claim, do not write it.
3. CLEAN TEXT ONLY: When populating the string fields in the schema, write natural, professional sentences. DO NOT include markdown formatting, bolding, hashtags (###), or label prefixes (like "Headline:" or "Body:") inside the actual text strings. 

### INPUT DATA
- Winner Strategy: {winner_name}
- Runner-Up Strategy: {runner_up_name}
- Loser Strategy: {loser_name}

- Pre-calculated Insights Payload:
{calculated_insights}
"""