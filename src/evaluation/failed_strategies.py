
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
import os
import logging
logger = logging.getLogger("app") 


def failed_strategies(state:AgentState):
    # if os.path.exists("src/data/failed_strategies.json"):
    #     data = load_state("failed_strategies")
    #     return {"failed_strategies" : data}
    chunk_status = {**state['chunk_status']}
    failed_strategies = []
    for strategy in chunk_status:
        if chunk_status == "Failed":
            failed_strategies.append(strategy)
    if len(failed_strategies) == 0:
        return {"failed_strategies" : "all strategies passed"}
    save_state(filename= "failed_strategies", data = failed_strategies)
    return {"failed_strategies" : failed_strategies}