from src.utils.save_state import save_state, load_state
from src.orchestration.agent_state import AgentState
import logging
logger = logging.getLogger("app") 
import os


def efficiency_scores(state:AgentState):
    # if os.path.exists("src/utils/data/efficiency_scores.json"):
    #     data = load_state("efficiency_scores")
    #     return {"efficiency_scores" : data}
    recall_precision_faithfullness_scores = state['recall_precision_faithfullness_scores']
    tokens_latency = state['tokens_latency']
    retrieved_chunks = state['retrieved_chunks']
    efficiency_scores = {}
    logger.debug(f"inside efficiency scores \n {recall_precision_faithfullness_scores}")
    for strategy in retrieved_chunks:
        context_recall = recall_precision_faithfullness_scores[strategy]['context_recall']
        faithfullness = recall_precision_faithfullness_scores[strategy]['faithfullness']
        total_tokens =  tokens_latency[strategy]["total_tokens"]
        latency = tokens_latency[strategy]["latency_seconds"]
        efficiency = (context_recall * faithfullness) / ((total_tokens * latency)/1000)

        efficiency_scores[strategy] = round(efficiency,2)
    
    logger.debug(f"Efficiency Scores Completed")
    save_state(filename= "efficiency_scores", data = efficiency_scores)
    return {"efficiency_scores" : efficiency_scores}
