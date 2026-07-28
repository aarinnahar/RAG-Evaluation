from src.orchestration.agent_state import AgentState
from src.models.models import embedding_model
from src.utils.save_state import save_state, load_state
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import logging

logger = logging.getLogger("app") 
model = embedding_model()

def proxy_recall_precision(state: AgentState):
    # if os.path.exists("src/data/vector_math_evaluator.json"):
    #     data = load_state("vector_math_evaluator")
    #     return {"proxy_scores" : data}
    
    logger.info("VECTOR MATH EVALUATOR STARTED")
    retrieved_chunks = state.get("retrieved_chunks", {})
    
    # Renamed the main dictionary to avoid shadowing
    final_metrics = {} 
    alpha = 0.6
    k = 4
    for strategy, data in retrieved_chunks.items():
        # Initialize lists for this strategy
        final_metrics[strategy] = {
            "max_similarity": [],
            "average_score": [],
            "coverage_score": [],
            "redundancy_score": [],
            "proxy_recall": [],
            "proxy_precision": []
        }
        
        for item in data:
            ans_embd = model.embed_query(item['answer'])
            
            # OPTIMIZATION: Embed all chunks ONCE
            chunks = [c.page_content for c in item['retrieved_chunks']] 
            chunk_vectors = model.embed_documents(chunks)
            
            # Calculate similarities between answer and chunks
            chunk_sims = []
            for chunk_vec in chunk_vectors:
                sim_score = cosine_similarity([ans_embd], [chunk_vec])[0][0]
                chunk_sims.append(sim_score)
            
            # 1. Max similarity & 2. Average Similarity & 3. Coverage
            max_sim = max(chunk_sims)
            avg_sim = np.mean(chunk_sims)
            coverage = sum(chunk_sims)
            
            final_metrics[strategy]["max_similarity"].append(max_sim)
            final_metrics[strategy]["average_score"].append(avg_sim)
            final_metrics[strategy]["coverage_score"].append(coverage)

            # 4. Redundancy (Dynamic calculation without hardcoding c1..c4)
            sim_matrix = cosine_similarity(chunk_vectors)
            np.fill_diagonal(sim_matrix, -1) # Your brilliant diagonal trick!
            
            # Finds the max similarity for each chunk, then averages them
            redundancy = np.mean(np.max(sim_matrix, axis=1)) 
            final_metrics[strategy]["redundancy_score"].append(redundancy)

            # 5. Proxy Recall & Precision
            proxy_recall = alpha * max_sim + (1-alpha) * coverage/ k
            inverse_redundancy = 1 - redundancy
            proxy_precision = inverse_redundancy + avg_sim
            
            final_metrics[strategy]["proxy_recall"].append(proxy_recall)
            final_metrics[strategy]["proxy_precision"].append(proxy_precision)

        # Aggregate the final scores for the strategy using dictionary comprehension
        for metric in final_metrics[strategy]:
            final_metrics[strategy][metric] = np.mean(final_metrics[strategy][metric])

    logger.info("proxy_recall_precision COMPLETE")
    logger.debug(f"Proxy Recall Precision Completed")
    save_state(filename= "proxy_recall_precision", data = final_metrics)
    return {"proxy_scores": final_metrics}