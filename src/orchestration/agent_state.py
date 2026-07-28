from typing import TypedDict, List, Dict, Any
from langchain_community.vectorstores import FAISS


class AgentState(TypedDict):
    file_path : str
    file_name : str
    chunk_size: int
    chunk_overlap: int
    chunk_factory : Dict[str, Any]
    score : Dict[str, Any]
    chunk_status : Dict[str, Any]
    text : str
    golden_dataset: List[Dict[str, str]]
    final_evaluation : Dict[str, Any]
    vectorstore: Dict[str, Any]
    
    failed_strategies : List[str]
    faithfullness_generator : dict[str, list]

    retrieved_chunks : dict[str, list]
    proxy_scores : dict[str, Any]

    recall_precision_faithfullness_scores : Dict[str, Any]
    tokens_latency : dict[str, Any]

    chunks_distribution : dict[str, Any]
    ingestion_time : dict[str, Any]

    efficiency_scores : dict[str, Any]

    combined_matrics : dict[str, Any]

    llm_json_payload : dict[str, Any]
    
    top_scorers : list

    llm_insights : dict[str, Any]