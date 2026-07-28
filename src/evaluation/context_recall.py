from src.orchestration.agent_state import AgentState
from src.models.models import embedding_model
from src.utils.save_state import save_state, load_state
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import os
import numpy as np
import logging

logger = logging.getLogger("app") 
model = embedding_model()

# step 1  converting answers to sentences



# step 2 convert sentences to embd



# step3 convert answer embd
# 
# 
# 
# step 4 compare answer and sentence embd
# 
# 
# 
# step 5 max

model = embedding_model()

def context_recall_mrr(state: AgentState):
    retrieved_chunks = {**state.get("retrieved_chunks", {})}
    context_recall_precision_scores = {} 
    threshold = 0.85

# STEP 1 — Split Answer Into Sentences &  STEP 2 — Embed Each Sentence & STEP 3 — Embed Retrieved Chunks & STEP 4 — Match Facts Against Chunks & 
    for strategy, data in retrieved_chunks.items():
        context_recall_precision_scores[strategy] = {
             "context_recall" : [],
             "context_precision" : [],
             "MRR"  : []
        }
        for item in data:
            answer = item['answer']
            sentences = sent_tokenize(answer)
            sent_embd = model.embed_documents(sentences)
            chunks = [c.page_content for c in item['retrieved_chunks']] 
            chunk_vectors = model.embed_documents(chunks)
            fact_covered = 0
            if len(sentences) == 0:
                context_recall_precision_scores[strategy]['context_recall'].append(0)
            # context recall
            for sent_vector in sent_embd: 
                sim_score = max(cosine_similarity([sent_vector], chunk_vectors)[0])
                if sim_score > threshold:
                    fact_covered += 1
            individual_context_recall = fact_covered/ len(sent_embd)
            context_recall_precision_scores[strategy]['context_recall'].append(individual_context_recall)
            
            # context precision
            if len(chunk_vectors) == 0:
                context_recall_precision_scores[strategy]['context_precision'].append(0)            
            else:
                covered = 0
                mrr_score = 0
                for index, retrieved_vectors in enumerate(chunk_vectors):
                    similarity_score = max(cosine_similarity([retrieved_vectors], sent_embd)[0])
                    if similarity_score > threshold:
                            covered += 1
                            if  mrr_score == 0:
                                mrr_score = 1 / (index + 1)
                                

                individual_context_precision = covered/len(chunk_vectors)
                context_recall_precision_scores[strategy]['context_precision'].append(individual_context_precision)
            
        context_recall_precision_scores[strategy]['context_recall'] = np.mean(context_recall_precision_scores[strategy]['context_recall'])
        context_recall_precision_scores[strategy]['context_precision'] = np.mean(context_recall_precision_scores[strategy]['context_precision'])
        context_recall_precision_scores[strategy]['MRR'] = np.mean(mrr_score)

    logger.debug(f"context_recall_MRR Completed")
    save_state(filename= "context_recall_MRR", data = context_recall_precision_scores)

    return {"context_recall_precision_mrr_scores" : context_recall_precision_scores}