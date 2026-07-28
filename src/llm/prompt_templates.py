from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field
from typing import Literal

class AnswerSchemaQueryIndependentEval(BaseModel):
    chunk_evalulation_score : int = Field(ge= 0, lt= 6, description= "Evaluate this chunk on the basis of given instructions in the prompt")

query_independent_template = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) pipeline. 
Your task is to evaluate the quality of a text chunk extracted from a document.
Rules;-

A high-quality chunk should be self-contained, easy to understand out of context, and focus on a cohesive topic. 

MUST RESPOND IN VALID JSON ONLY INTEGER VALUES

TEXT CHUNK:
"{chunk_text}"

Evaluate this chunk on a scale of 1 to 5 based on its Standalone Comprehensibility.
1 = Incomprehensible out of context (e.g., starts mid-sentence, dangling pronouns).
5 = Perfectly self-contained and clear.

Return ONLY the integer score."""

parser = PydanticOutputParser(pydantic_object=AnswerSchemaQueryIndependentEval) 
format_instructions = parser.get_format_instructions()

query_independent_prompt = PromptTemplate(template= query_independent_template,
                                          input_variables= ["chunk_text"],
                                          partial_variables={"format_instruction": format_instructions})

