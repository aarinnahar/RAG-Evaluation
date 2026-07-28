
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
import os
import logging
logger = logging.getLogger("app") 

def calculate_junk_ratio(chunk: str) -> float:
    """Calculates the ratio of alphanumeric characters out of total characters in the chunk."""
    if not chunk:
        return 0.0  # Exit immediately to prevent ZeroDivisionError

    # Removed the [] inside sum() to use a memory-efficient generator instead of a list
    alpha_numeric = sum(c.isalnum() for c in chunk)
    total_char = len(chunk)

    score = alpha_numeric / total_char
    return score


def sentence_boundary_score(chunk: str) -> float:
    """Checks if the chunk starts with a capital letter and ends with terminal punctuation."""
    if not chunk:
        return 0.0

    score = 0.0
    # Strip removes any accidental leading/trailing spaces or newlines from the chunk
    clean_chunk = chunk.strip() 

    if not clean_chunk:
        return 0.0

    # +0.5 if it starts with a capital letter
    if clean_chunk[0].isupper():
        score += 0.5
        
    # +0.5 if it ends with terminal punctuation
    if clean_chunk[-1] in ".?!":
        score += 0.5

    return score 


# ------------------------FINAL PROXY EVALUATOR-----------------------------

def evaluate_chunks_set(state: AgentState):
    """Aggregates all proxy metrics to return a single normalized score between 0.0 and 1.0"""
    # if os.path.exists("src/data/heuristic_evaluation.json"):
    #     data = load_state("heuristic_evaluation")
    #     return {"score" : data}
    
    logger.info("CHUNK SET EVALUATION STARTED")
    chunk_factory = {**state["chunk_factory"]}
    strategy_scores = {}
    if not chunk_factory:
        return 0.0  # Exit immediately
    for strategy in chunk_factory:
        chunks = chunk_factory[strategy]
        # Add the metrics and divide by 2 to keep the max possible score at 1.0
        scores = [(calculate_junk_ratio(chunk) + sentence_boundary_score(chunk)) / 2 for chunk in chunks]
        average_score = sum(scores) / len(scores)
        strategy_scores[strategy] = average_score
    logger.info(f"strategy scores : {strategy_scores}")
    
    logger.debug(f"Heuristic Evaluation Completed")
    save_state(filename= "heuristic_evaluation", data = strategy_scores)
    return {"score" :strategy_scores}




