from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from src.llm.llm_client import get_llm
from collections import defaultdict
import numpy as np
import time
import os
import logging
logger = logging.getLogger("app") 
import json

class FaithfullnessSchema(BaseModel):
    answer : str = Field(description= "evaluate the chunk and give it a score")

def faithfullness_generator(state : AgentState):
    if os.path.exists("src/utils/data/faithfullness_generator.json"):
        logger.info("Skipping faithfullness_generator")
        if os.path.exists("src/utils/data/tokens_latency.json"):
            with open("src/utils/data/faithfullness_generator.json") as f1:
                with open("src/utils/data/tokens_latency.json") as f2:
                    return {"faithfullness_generator" : json.load(f1),"tokens_latency" : json.load(f2)}
    retrieved_chunks = {**state['retrieved_chunks']}
    answers = {}
    tokens_latency = {}
    template = """You are an expert, precise assistant. Your only task is to answer the user's question using STRICTLY the information provided in the context below.

Rules:
1. You must not use any outside knowledge or pre-trained information.
2. If the context contains the answer, synthesize it clearly and concisely.
3. If the provided context does not contain sufficient information to answer the question, you must not guess. Instead, output exactly: "INSUFFICIENT_CONTEXT"
Question: {question}
Context:
{retrieved_chunks}
    """

    prompt = PromptTemplate(template= template, input_variables= ['question', 'retrieved_chunks'])
    llm = get_llm()
    structured_llm = llm.with_structured_output(schema=FaithfullnessSchema, include_raw= True)
    for strategy, data in retrieved_chunks.items():
        answers[strategy] = []
        tokens_latency[strategy] = {"total_tokens" : [],
                                    "input_tokens" : [],
                                    "output_tokens" : [],
                                    "latency_seconds" : []}
        logger.info(f"LLM faithfullness_generator NODE STARTS : {strategy}")
        for item in data:
# ... inside your item loop ...
            query = item['question'] 
            chunks_texts = [c.page_content for c in item['retrieved_chunks']]
            chunks_context = "\n\n---NEXT CHUNK---\n\n".join(chunks_texts) 
            main_prompt = prompt.format(question=query, retrieved_chunks=chunks_context)
            
            # The message starts as the main prompt
            message = main_prompt 
            
            for trial in range(3):
                logger.info(f'Trial run {trial + 1} for {query}')
                start = time.time()
                response = structured_llm.invoke(message)
                end = time.time()
                total_time = end - start
                if response['parsing_error']:
                    logger.warning(f"Parsing error on trial {trial + 1}: {response['parsing_error']}")
                    # Append the error TO THE MAIN PROMPT so it remembers the rules
                    message = main_prompt + f"\n\nSYSTEM WARNING - FIX PREVIOUS ERROR: {response['parsing_error']}"
                else:
                    input_tokens = response['raw'].usage_metadata["input_tokens"]
                    output_tokens = response['raw'].usage_metadata["output_tokens"]
                    total_tokens = response['raw'].usage_metadata["total_tokens"]
                    total_latency = item["vector_db_search"] +  total_time
                    data_item = {**item,
                                 "llm_answer" :response['parsed'].answer,
                                 "metrics" : {"total_tokens" : total_tokens,
                                              "input_tokens" : input_tokens,
                                              "output_tokens" : output_tokens,
                                              "latency_seconds" : total_latency}} 
                    tokens_latency[strategy]["total_tokens"].append(total_tokens)
                    tokens_latency[strategy]["input_tokens"].append(input_tokens)
                    tokens_latency[strategy]["output_tokens"].append(output_tokens)
                    tokens_latency[strategy]["latency_seconds"].append(total_latency)
                    answers[strategy].append(data_item)
                    
                    break
            else:
                # This 'else' triggers ONLY if the loop finishes without a 'break' (meaning it failed 3 times)
                logger.error(f"Failed to get a valid score for query: '{query}' after 3 attempts. Assigning 0.")
    
        tokens_latency[strategy]["total_tokens"]= np.mean(tokens_latency[strategy]["total_tokens"])
        tokens_latency[strategy]["latency_seconds"]= np.mean(tokens_latency[strategy]["latency_seconds"])
    
    logger.info(f"LLM EVALUATION FINISHED!")
    logger.debug(f"Faithfullness Generator Completed")
    save_state(filename= "faithfullness_generator", data = answers)
    save_state(filename= "tokens_latency", data = tokens_latency)
    return {"faithfullness_generator" : answers, "tokens_latency" : tokens_latency}










