from src.chunking.character_text_splitter import character_text_splitter_chunker
from src.chunking.recursive_character_text_splitter import recursive_character_text_splitter_chunker
from src.chunking.token_text_splitter import token_text_splitter_chunker
from src.chunking.semantic_chunking import semantic_chunking_pro
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
import time
import os
import logging
logger = logging.getLogger("app") 

def chunking_factory(state:AgentState):
    # if os.path.exists("src/data/chunk_factory.json"):
    #     if os.path.exists("src/data/total_ingestion_time.json"):
    #         with open("src/data/chunk_factory.json") as f1:
    #             data = load_state(f1)
    #             with open("src/data/total_ingestion_time.json") as f2:
    #                 ingest = load_state(f2)
    #         return {"chunk_factory" : data,"ingestion_time" : ingest}
    text = state["text"]
    chunk_size = state['chunk_size']
    chunk_overlap = state['chunk_overlap']
    
    chunk_factory = {
        "character_text_splitter_chunks" : [],
        "recursive_text_splitter_chunks" : [],
        "token_text_splitter_chunks" : [],
        "semantic_chunks" : []
    }
    character_splitter_start = time.time()
    chunk_factory['character_text_splitter_chunks'] = character_text_splitter_chunker(text, chunk_size, chunk_overlap)
    character_splitter_end = time.time()
    diff_chara = character_splitter_end - character_splitter_start

    recursive_character_splitter_start = time.time()
    chunk_factory['recursive_text_splitter_chunks'] = recursive_character_text_splitter_chunker(text, chunk_size, chunk_overlap)
    recursive_character_splitter_end = time.time()
    diff_recur = recursive_character_splitter_end - recursive_character_splitter_start
    
    token_text_splitter_start = time.time()
    chunk_factory["token_text_splitter_chunks"] = token_text_splitter_chunker(text, chunk_size, chunk_overlap)
    token_text_splitter_end = time.time()
    diff_token = token_text_splitter_end - token_text_splitter_start
    
    semantic_splitter_start = time.time()
    chunk_factory["semantic_chunks"] = semantic_chunking_pro(text)
    semantic_splitter_end = time.time()
    diff_semantic = semantic_splitter_end - semantic_splitter_start
    
    ingestion = {
        "character_text_splitter_chunks" : diff_chara,
        "recursive_text_splitter_chunks" : diff_recur,
        "token_text_splitter_chunks" : diff_token,
        "semantic_chunks" : diff_semantic
    }
    save_state(filename= "chunk_factory", data = chunk_factory)
    save_state(filename= "ingestion_time", data = ingestion)
    logger.debug(f"Chunking Factory Completed")
    return {"chunk_factory" : chunk_factory, "ingestion_time" : ingestion}
