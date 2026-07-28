from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state
import numpy as np
import logging
logger = logging.getLogger("app") 

def combined_metrics_node(state: AgentState):
  
    recall_precision_faithfullness_scores = state['recall_precision_faithfullness_scores']

    tokens_latency = state['tokens_latency']
    ingestion_time = state['ingestion_time']
    
    combined_matrics = {}

    
    for strategy, data in recall_precision_faithfullness_scores.items():
        
        combined_matrics[strategy] = data
    
    for strategy, data in tokens_latency.items():
        premium_cost = 100*(((np.mean(data['input_tokens']) / 1000000) * 5) + ((np.mean(data['output_tokens']) / 1000000) * 15))
        fast_cost = 100*(((np.mean(data['input_tokens']) / 1000000) * 0.15) + ((np.mean(data['output_tokens']) / 1000000) * 0.60))
        open_cost = 100*(((np.mean(data['input_tokens']) / 1000000) * 0.05) + ((np.mean(data['output_tokens']) / 1000000) * 0.10))
        
        combined_matrics[strategy]['premium_cost'] =  premium_cost
        combined_matrics[strategy]['fast_cost'] =  fast_cost
        combined_matrics[strategy]['open_cost'] =  open_cost
        combined_matrics[strategy] = combined_matrics[strategy] | data

    for strategy in ingestion_time:
        combined_matrics[strategy].update({"ingestion_time": ingestion_time[strategy]})

    updated_matrics = {}
    for strategy, data in combined_matrics.items():
        updated_matrics.setdefault(strategy,{})
        for i in data:
            updated_matrics[strategy][i] = round(np.mean(data[i]), 3)
    
    
    logger.debug(f"this is combined metrics {updated_matrics}")
    logger.debug(f"combined metrics completed successfully")
    save_state(filename= "combined_matrics", data = updated_matrics)
    return {"combined_matrics" : updated_matrics}



















    results = state["strategy_results"]
    flags = []
    
    # 1. Look at Fixed-Size 512 vs Fixed-Size 1024 (Diminishing Returns Check)
    f512 = results.get("fixed_512", {})
    f1024 = results.get("fixed_1024", {})
    
    if f512 and f1024:
        recall_gain = f1024["llm_eval_score"] - f512["llm_eval_score"]
        token_increase = f1024["prompt_tokens"] / f512["prompt_tokens"]
        
        if recall_gain < 0.05 and token_increase > 1.8:
            flags.append(
                f"CRITICAL: Fixed-1024 increased recall by only {recall_gain*100:.1f}%, "
                f"but inflated token consumption by {token_increase:.1f}x. Wastes money."
            )
            
    # 2. Check alignment between Proxy Recall and LLM Eval Score
    for strategy, metrics in results.items():
        delta = metrics["proxy_recall"] - metrics["llm_eval_score"]
        if delta > 0.30:
            flags.append(
                f"WARNING: {strategy} has high keyword match (Proxy Recall) but low LLM comprehension. "
                f"Text is likely cut off mid-sentence, destroying semantic context."
            )

    # 3. Determine the absolute winner using your math formula
    winner = max(results, key=lambda k: (results[k]["llm_eval_score"] / results[k]["prompt_tokens"]))

    return {
        "calculated_winner": winner,
        "efficiency_flags": flags
    }