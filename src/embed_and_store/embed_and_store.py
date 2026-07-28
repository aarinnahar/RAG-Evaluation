from src.models.models import embedding_model
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore
import faiss
import time
import numpy as np
import logging
logger = logging.getLogger("app") 

def embed_and_store(state : AgentState):
    model = embedding_model()
    chunk_status = {**state['chunk_status']}
    ingestion_time = {**state['ingestion_time']}
    chunk_factory = {**state['chunk_factory']}
    vectorstore = {}
    logger.info(f"chunk_status is {chunk_status}")
    for strategy, chunks in chunk_factory.items():
        logger.info(f"Embed & Store started for {strategy}")

        if chunk_status[strategy] == "Passed":
            start_time = time.time()
            vectors = model.embed_documents(chunks)
            vectors = np.array(vectors).astype('float32')
            dimension = vectors.shape[1]  
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)

            document = [Document(page_content = chunk)for chunk in chunks]

            docstore = InMemoryDocstore({
            str(i): doc for i, doc in enumerate(document)})

            index_to_docstore_id = {
            i: str(i) for i in range(len(document))}

            vectorstore[strategy] = FAISS(
            embedding_function=model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id)

            end_time = time.time()

            total = end_time - start_time
            
            total_ingestion_time =  ingestion_time[strategy] + total

            ingestion_time[strategy] = total_ingestion_time
    logger.debug(f"Embed & Store Completed")
    save_state(filename= "total_ingestion_time", data = ingestion_time)
    return {"vectorstore" : vectorstore, "ingestion_time" : ingestion_time}










#     function
# 🧠 How next node uses it

# In next node:

# retriever = vectorstore.as_retriever()

# Then:

# docs = retriever.get_relevant_documents("your query")