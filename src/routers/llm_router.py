from src.orchestration.agent_state import AgentState
import logging
logger = logging.getLogger("app") 

def llm_router(state:AgentState):
    logger.info(f"inside LLM Router Node")
    chunk_status = state["chunk_status"]
    for strategy in chunk_status:
        if chunk_status[strategy] == "Passed":
            logger.info(f"Routing to Embed & Store")
            return "embed_and_store"
        else:
            return "failed_strategies"
    