import logging
import asyncio
from src.utils.logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from src.orchestration.agent_state import AgentState

from src.ingestion.text_loader import text_loader
from src.ingestion.golden_dataset_loader import golden_dataset_loader
from src.chunking.chunking_factory import chunking_factory
from src.evaluation.heuristic import evaluate_chunks_set
from src.gating.decision_gate import decision_gate
from src.evaluation.failed_strategies import failed_strategies
from src.embed_and_store.embed_and_store import embed_and_store
from src.retrieval.retireval import retrieval
from src.evaluation.proxy_recall_precision import proxy_recall_precision
from src.evaluation.faithfullness_generator import faithfullness_generator
from src.evaluation.recall_precision_faithfullness_evaluator import recall_precision_faithfullness_evaluator
from src.evaluation.chunk_distribution import calculate_chunk_distribution
from src.evaluation.efficiency_score import efficiency_scores
from src.results.combined_matrics import combined_metrics_node
from src.results.analyse_matrics import analyze_metrics_node
from src.results.llm_insights_generator import llm_insights_generator
from src.report_design.graphs import create_graphs
from src.results.result_report_generator import result_report_designer

graph = StateGraph(AgentState)

graph.add_node("text_loader", text_loader)
# graph.add_node("golden_dataset_loader", golden_dataset_loader)
graph.add_node("chunking_factory", chunking_factory)
graph.add_node("heuristic_evaluation", evaluate_chunks_set)
graph.add_node("decision_gate", decision_gate)
graph.add_node("failed_strategies", failed_strategies)
graph.add_node("embed_and_store", embed_and_store)
graph.add_node("retrieval", retrieval)
graph.add_node("proxy_recall_precision", proxy_recall_precision)
graph.add_node("faithfullness_generator", faithfullness_generator)
graph.add_node("recall_precision_faithfullness_evaluator", recall_precision_faithfullness_evaluator)
graph.add_node("calculate_chunk_distribution", calculate_chunk_distribution)
graph.add_node("efficiency_scores", efficiency_scores)
graph.add_node("combined_metrics_node", combined_metrics_node)
graph.add_node("analyze_metrics_node", analyze_metrics_node)
graph.add_node("llm_insights_generator", llm_insights_generator)
graph.add_node("create_graphs", create_graphs)
graph.add_node("result_report_designer", result_report_designer)

graph.add_edge(START,"text_loader")
# graph.add_edge("text_loader", "golden_dataset_loader")
graph.add_edge("text_loader","chunking_factory")
graph.add_edge("chunking_factory","heuristic_evaluation")
graph.add_edge("heuristic_evaluation","decision_gate")
graph.add_edge("decision_gate","failed_strategies")
graph.add_edge("failed_strategies","embed_and_store")
graph.add_edge("embed_and_store","retrieval")
graph.add_edge("retrieval","proxy_recall_precision")
graph.add_edge("proxy_recall_precision","faithfullness_generator")
graph.add_edge("faithfullness_generator","recall_precision_faithfullness_evaluator")
graph.add_edge("recall_precision_faithfullness_evaluator","calculate_chunk_distribution")
graph.add_edge("calculate_chunk_distribution","efficiency_scores")
graph.add_edge("efficiency_scores","combined_metrics_node")
graph.add_edge("combined_metrics_node","analyze_metrics_node")
graph.add_edge("analyze_metrics_node","llm_insights_generator")
graph.add_edge("llm_insights_generator","create_graphs")
graph.add_edge("create_graphs","result_report_designer")

graph.add_edge("result_report_designer", END)

workflow = graph.compile()


# async def main():
#     return await workflow.ainvoke({"input": "hello"})

# result = asyncio.run(main())
