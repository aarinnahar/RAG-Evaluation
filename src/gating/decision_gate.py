from src.orchestration.agent_state import AgentState
from src.config.settings import Settings
import logging
logger = logging.getLogger("app") 

def decision_gate(state: AgentState):
    logger.info("DECISION GATE STARTED")
    scores = {**state['score']}
    chunk_status = {}
    setting = Settings()
    THRESHOLD = setting.chunk_threshold
    for strategy in scores:
        if scores[strategy] > THRESHOLD:
            logger.info(f"PASS: Proxy score {scores[strategy]:.2f} STRATEGY : {strategy} meets threshold {THRESHOLD}. Proceeding to LLM.")        
            chunk_status[strategy] = "Passed"
        else:
            logger.warning(f" FAIL: Proxy score {scores[strategy]:.2f} STRATEGY : {strategy} is below threshold {THRESHOLD}. Halting pipeline.")
            chunk_status[strategy] = "Failed"
    logger.debug(f"Decision Gate Completed")
    return {"chunk_status" : chunk_status}
    

