from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
import os
import logging
logger = logging.getLogger("app") 
import time


# golden load 
# loop through each item 
# question embed
# vector store retrieve
# question + retrieved chunks LLM for evaluation 
# scores reveice and store for all questions
# aggregation

def retrieval(state : AgentState):
    # if os.path.exists("src/data/retrieved_chunks.json"):
    #     data = load_state("retrieved_chunks")
    #     return {"retrieved_chunks" : data}
    vectorstore = {**state["vectorstore"]}
    k = 4
    golden_dataset = state['golden_dataset']
    retrieved_chunks = {}
    for strategy in vectorstore:
        retrieved_chunks.setdefault(strategy, [])
        logger.info(f"RETRIEVER NODE STARTS EVALUATION strategy : {strategy}")
        for item in golden_dataset:
            query = item['query'] 
            answer = item['answer'] 
            retriever = vectorstore[strategy].as_retriever(search_kwargs = {"k":k})
            start = time.time()
            chunks = retriever.invoke(query)
            end = time.time()
            logger.info(f"Query is {query} \n Strategy is {strategy} total chunks retrieved {len(chunks)}")
            total_time = end - start 
            data = {'question' : query,
                    "answer" : answer,
                    "vector_db_search" : total_time,
                    "retrieved_chunks" : chunks}
            retrieved_chunks[strategy].append(data)


    logger.info(f"RETRIEVAL FINISHED!")
    save_state(filename= "retrieved_chunks", data = retrieved_chunks)
    logger.debug(f"Retrievel Completed")
    return {"retrieved_chunks" : retrieved_chunks}










