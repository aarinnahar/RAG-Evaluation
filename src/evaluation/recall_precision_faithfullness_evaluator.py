from src.orchestration.agent_state import AgentState
from src.models.schemas.context_recall_precision_faithfullness_schema import RagEvaluationAudit
from src.prompt_templates.context_precision_faithfullness_evaluator import CONTEXT_RECALL_PRECISION_FAITHFULLNESS
from src.utils.save_state import save_state, load_state
from langchain_core.prompts import PromptTemplate
from src.llm.llm_client import get_llm
import numpy as np
from pathlib import Path
import os
import logging
logger = logging.getLogger("app") 
import json
script_dir = Path(__file__).resolve().parent          # current dir
parent_dir = script_dir.parent   


def recall_precision_faithfullness_evaluator(state: AgentState):
    logger.debug(f"path is {script_dir}")
    comp_path = parent_dir / 'utils/data/faithfullness_relevancy_scores.json'
    logger.debug(f"complete path is {comp_path}")

    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            data = json.load(f)
        logger.debug("Loaded Successfully !!!")
        logger.debug(f"this is data{data}")
        return {"recall_precision_faithfullness_scores" : data}
    # Shallow copy the generator dict
    faithfullness_generator = {**state['faithfullness_generator']}
    
    scores = {}
    recall_precision_faithfullness_scores = {}

    prompt = PromptTemplate(
        template=CONTEXT_RECALL_PRECISION_FAITHFULLNESS, 
        input_variables=['question', 'generated_answer', 'retrieved_chunks', 'golden_answer']
    )
    llm = get_llm()
    structured_llm = llm.with_structured_output(schema=RagEvaluationAudit, include_raw=True)
    
    for strategy, data in faithfullness_generator.items():
        # Standardized on 'faithfullness' internally for consistency
        scores[strategy] = {"faithfullness": [], "context_recall": [], "context_precision": []}
        
        logger.info(f"LLM context_recall_faithfullness_evaluator NODE STARTS : {strategy}")
        
        for item in data:
            generated_answer = item['llm_answer'] 
            question = item['question']
            golden_answer = item['answer']
            
            chunks_texts = [c.page_content if hasattr(c, 'page_content') else c for c in item['retrieved_chunks']]
            chunks_context = "\n\n---NEXT CHUNK---\n\n".join(chunks_texts) 
            
            main_prompt = prompt.format(
                question=question, 
                generated_answer=generated_answer, 
                retrieved_chunks=chunks_context, 
                golden_answer=golden_answer
            )
            message = main_prompt             
            for trial in range(3):
                response = structured_llm.invoke(message)
                
                if response['parsing_error']:
                    logger.warning(f"Parsing error on trial {trial + 1}: {response['parsing_error']}")
                    message = main_prompt + f"\n\nSYSTEM WARNING - FIX PREVIOUS ERROR: {response['parsing_error']}"
                else:
                    parsed_payload = response['parsed']
                    
                    # --- 1. CALCULATE ROW-LEVEL CONTEXT RECALL ---
                    recall_facts = parsed_payload.context_recall_analysis
                    if recall_facts:
                        true_facts = sum(1 for fact in recall_facts if fact.found_in_chunks)
                        row_recall = true_facts / len(recall_facts)
                    else:
                        row_recall = 0.0
                    scores[strategy]["context_recall"].append(row_recall)
                    
                    # --- 2. CALCULATE ROW-LEVEL CONTEXT PRECISION (Mean Average Precision) ---
                    precision_chunks = parsed_payload.context_precision_analysis
                    # Sort by chunk_index to ensure we process in true retrieval order
                    precision_chunks = sorted(precision_chunks, key=lambda x: x.chunk_index)
                    
                    relevant_count = 0
                    running_precision_sum = 0.0
                    for rank, chunk in enumerate(precision_chunks, start=1):
                        if chunk.is_relevant_to_question:
                            relevant_count += 1
                            running_precision_sum += (relevant_count / rank)
                            
                    row_precision = running_precision_sum / relevant_count if relevant_count > 0 else 0.0
                    scores[strategy]["context_precision"].append(row_precision)
                    
                    # --- 3. CALCULATE ROW-LEVEL faithfullness ---
                    faith_statements = parsed_payload.faithfullness_analysis
                    if faith_statements:
                        true_statements = sum(1 for stmt in faith_statements if stmt.supported_by_chunks)
                        row_faithness = true_statements / len(faith_statements)
                    else:
                        row_faithness = 0.0
                    scores[strategy]["faithfullness"].append(row_faithness)
                        
                    break
            else:
                logger.error(f"Failed to get a valid score for generated_answer: '{generated_answer}' after 3 attempts.")
                # Append fallback 0s to maintain parallel array structures across failures
                scores[strategy]["context_recall"].append(0.0)
                scores[strategy]["context_precision"].append(0.0)
                scores[strategy]["faithfullness"].append(0.0)
    
    # --- MACRO AGGREGATION LAYER ---
    for strategy, data in scores.items():
        recall_precision_faithfullness_scores[strategy] = {}
    
        # Protect against division by zero errors on empty lists
        context_recall = sum(data['context_recall']) / len(data['context_recall']) if data['context_recall'] else 0.0
        context_precision = sum(data['context_precision']) / len(data['context_precision']) if data['context_precision'] else 0.0
        faithfullness = sum(data['faithfullness']) / len(data['faithfullness']) if data['faithfullness'] else 0.0

        # Output matches your UI key structure ("faithfullness")
        recall_precision_faithfullness_scores[strategy] = {
            "context_recall": context_recall,
            "context_precision": context_precision,
            "faithfullness": faithfullness
        }

    logger.debug(f"Context_Recall_Faithfullness Evaluation Completed")
    logger.info("LLM Context_Recall_Faithfullness EVALUATION FINISHED!")
    save_state(filename="faithfullness_relevancy_scores", data=recall_precision_faithfullness_scores)
    
    return {"recall_precision_faithfullness_scores": recall_precision_faithfullness_scores}





