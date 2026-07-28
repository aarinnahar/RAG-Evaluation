from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state
import logging
logger = logging.getLogger("app") 

def analyze_metrics_node(state: AgentState):
    efficiency_scores = state['efficiency_scores']
    combined_matrics = {**state['combined_matrics']}
    top_scorers = sorted(efficiency_scores.items(), key = lambda item : item[1], reverse= True)
    winner = top_scorers[0][0]
    runner_up = top_scorers[1][0]
    loser = top_scorers[3][0]
    llm_payload = {}
    llm_payload.setdefault("winner_vs_runnerup_insights", [])

    calculated_insights = {"retrieval_insights" : [],
                           "generation_insights" : [],
                           "efficiency_insights" : []} 

    accuracy_scores ={}

    winner_latency = combined_matrics[winner]['latency_seconds']
    runner_up_latency = combined_matrics[runner_up]['latency_seconds']
    for strategy, data in combined_matrics.items():
        score = (
            (0.4 * data["faithfullness"]) +
            (0.4 * data["context_recall"]) +
            (0.2 * data["context_precision"]) 
        )
        accuracy = round(score * 100, 1)
        accuracy_scores[strategy] = accuracy

    for strategy in accuracy_scores:
        combined_matrics[strategy].update({"overall_accuracy" :accuracy_scores[strategy]})

    percentage_recall  = ((combined_matrics[winner]["context_recall"] - combined_matrics[runner_up]["context_recall"]) / combined_matrics[winner]["context_recall"]) * 100
    percentage_precision  = ((combined_matrics[winner]["context_precision"] - combined_matrics[runner_up]["context_precision"]) / combined_matrics[winner]["context_precision"]) * 100
    percentage_faithfullness  = ((combined_matrics[winner]["faithfullness"] - combined_matrics[runner_up]["faithfullness"]) / combined_matrics[winner]["faithfullness"]) * 100
    percentage_efficiency  = ((efficiency_scores[winner] - efficiency_scores[runner_up]) / efficiency_scores[winner]) * 100
    percentage_total_tokens  = ((combined_matrics[winner]["total_tokens"] - combined_matrics[runner_up]["total_tokens"]) / combined_matrics[winner]["total_tokens"]) * 100
    percentage_latency_seconds  = ((combined_matrics[winner]["latency_seconds"] - combined_matrics[runner_up]["latency_seconds"]) / combined_matrics[winner]["latency_seconds"]) * 100
    percentage_ingestion_time  = ((combined_matrics[winner]["ingestion_time"] - combined_matrics[runner_up]["ingestion_time"]) / combined_matrics[winner]["ingestion_time"]) * 100

    mark = ''
  # Efficiency Score (Value for Money)
    if -5 <= percentage_efficiency <= 5:
        mark = f"TIE: Both {winner} and {runner_up} offer a nearly identical balance of quality vs. operational cost."
    elif percentage_efficiency > 5:
        # You need to check WHY it won to give an accurate insight
        if winner_latency < runner_up_latency:
            mark = f"SPEED AND VALUE: {winner} is the smarter production choice. Its extreme speed and cost efficiency outscore {runner_up} by {percentage_efficiency:.1f}%."
        else:
            mark = f"PREMIUM VALUE CONFIRMED: {winner} wins outright. Its overwhelmingly high retrieval quality easily justifies its slower processing speed, outscoring {runner_up}'s efficiency by {percentage_efficiency:.1f}%."
    calculated_insights["efficiency_insights"].append(mark)

# FAITHFULLNESS((Safety & Hallucination Check))
    if -5 <= percentage_faithfullness <= 5:
        mark = f"TIE: Both strategies {winner} and {runner_up} are equally safe and exhibit the same level of factual honesty."
    elif percentage_faithfullness > 5:
        # Positive means winner did better, so praise the winner
        mark = f"EDGE: {winner} is more honest. {runner_up} has a minor ({percentage_faithfullness:.1f}%) risk of introducing loose interpretations or minor hallucinations."
    elif percentage_faithfullness < -5:  
        # Negative means runner-up did better, so praise the runner-up
        mark = f"CRITICAL ADVANTAGE: {runner_up} strictly adheres to the PDF. {winner} has a massive ({abs(percentage_faithfullness):.1f}%) failure in faithfulness and frequently makes up answers out of thin air."
    calculated_insights["generation_insights"].append(mark)


# CONTEXT_RECALL(Memory Check)
    if -5 <= percentage_recall <= 5:
        mark = f"TIE: Both strategies {winner} and {runner_up} have an identical memory span and find the exact same number of correct answers."
    
    elif percentage_recall > 5:
        # Positive means winner did better. Praise the winner.
        mark = f"EDGE: {winner} finds slightly more source details ({percentage_recall:.1f}% higher recall). {runner_up} occasionally leaves out minor supporting context."
    
    elif percentage_recall < -15:  
        # Highly negative means runner-up did MUCH better. The winner failed badly here.
        mark = f"MAJOR QUALITY GAP: {winner} suffers from severe information loss ({abs(percentage_recall):.1f}% lower recall). It completely misses key blocks of information that {runner_up} easily uncovers."
    
    elif percentage_recall < -5:  
        # Slightly negative means runner-up did somewhat better.
        mark = f"QUALITY GAP: {winner} has noticeable information loss ({abs(percentage_recall):.1f}% lower recall) compared to {runner_up}, missing some relevant context."
    
    calculated_insights["retrieval_insights"].append(mark)

    # # Answer Relevancy (Focus Check)
    # if percentage_relevancy < 5:
    #     mark = f"TIE: Both strategies {winner} and {runner_up} produce answers that directly address the user's question without rambling."
    # else:  # gap >= 5% (Using a tighter 5% threshold here since answers should always be relevant)
    #     mark = f"RELEVANCY GAP:{runner_up}'s final answers are {percentage_relevancy}% less focused on the prompt, occasionally drifting into unrelated topics compared to {winner}."
    # calculated_insights.append(mark)    
    
   
    # Context Precision (Fluff Detector)    
    if -5 <= percentage_precision <= 5:
        mark = f"TIE: Both strategies {winner} and {runner_up} extract data cleanly with equal focus."
    
    elif percentage_precision > 5:
        # Positive means winner did better (higher precision). The runner-up has the fluff.
        mark = f"MARGINAL FLUFF: {winner} is more precise. {runner_up}'s chunks contain {percentage_precision:.1f}% more unrelated surrounding text, slightly diluting the core answer."
    
    elif percentage_precision < -5:  
        # Negative means runner-up did better. The winner has the fluff.
        mark = f"NOISE WARNING: {winner} pulls in a massive amount of junk text fluff ({abs(percentage_precision):.1f}% lower precision). This unnecessarily floods the context window with useless information compared to {runner_up}."
    
    calculated_insights["retrieval_insights"].append(mark)
    

    # # MRR (Ranking Accuracy)
    # if percentage_MRR < 5:
    #     mark = f"TIE: Both strategies {winner} and {runner_up} put the absolute best chunk directly at the top of the search results."
    # else:  # gap >= 5%
    #     mark = f"RANKING DELAY: {runner_up} is less precise at ranking. The exact answer chunk is buried deeper down the list by {percentage_MRR}% compared to {winner}."
    # calculated_insights.append(mark)
    
    # Tokens Usage (The Cost Driver)
    if -5 <= percentage_total_tokens <= 5:
        mark = f"TIE: Cost profiles are identical; both {winner} and {runner_up} consume roughly the same volume of tokens per query."
    
    elif percentage_total_tokens > 5:
        # Positive means winner uses MORE tokens. Therefore, the runner-up is cheaper.
        mark = f"COST DIFFERENCE: {runner_up} uses {percentage_total_tokens:.1f}% fewer tokens per query compared to {winner}, offering mild financial savings over time."
    
    elif percentage_total_tokens < -5:
        # Negative means runner-up uses MORE tokens. Therefore, the winner is cheaper.
        mark = f"COST SAVINGS: {winner} is incredibly resource-efficient, cutting token usage by {abs(percentage_total_tokens):.1f}% compared to {runner_up}. If you run a high-volume app, {winner} will dramatically lower your API bills."
    
    calculated_insights["efficiency_insights"].append(mark)

    # Latency (User Wait Time)
    if -5 <= percentage_latency_seconds <= 5:
        mark = f"TIE: Users will experience the exact same response speeds across both {winner} and {runner_up} strategies."
    elif percentage_latency_seconds > 5:
        mark = f"SPEED EDGE: {winner} responds slightly faster ({percentage_latency_seconds:.1f}% lower latency), shaving off a noticeable fraction of a second."
    elif percentage_latency_seconds < -5:
        mark = f"PERFORMANCE BOTTLENECK: {winner} causes major processing lag. {runner_up} is {abs(percentage_latency_seconds):.1f}% faster, making it vastly superior for real-time applications."
    calculated_insights["efficiency_insights"].append(mark)

   # Total Ingestion Time (PDF Processing Speed)
    if -5 <= percentage_ingestion_time <= 5:
        mark = f"TIE: Both chunking strategies {winner} and {runner_up} take the same time to initially parse and embed the PDF files."
    
    elif percentage_ingestion_time > 5:
        # Positive means the winner is faster
        mark = f"INGESTION SPEED: {winner} processes and chunks the raw document files {percentage_ingestion_time:.1f}% faster than {runner_up}."
        
    elif percentage_ingestion_time < -5:
        # Negative means the runner-up is faster. Use abs() to remove the negative sign!
        mark = f"INGESTION SPEED: {runner_up} processes and chunks the raw document files {abs(percentage_ingestion_time):.1f}% faster than {winner}."
    
    calculated_insights["efficiency_insights"].append(mark)

 

    # Use a small epsilon to prevent division by zero errors
    eps = 1e-9 

    loser_gaps = {
        "context_recall": (abs(combined_matrics[winner]["context_recall"] - combined_matrics[loser]["context_recall"]) / (combined_matrics[winner]["context_recall"] + eps)) * 100,
        "context_precision": (abs(combined_matrics[winner]["context_precision"] - combined_matrics[loser]["context_precision"]) / (combined_matrics[winner]["context_precision"] + eps)) * 100,
        "faithfullness": (abs(combined_matrics[winner]["faithfullness"] - combined_matrics[loser]["faithfullness"]) / (combined_matrics[winner]["faithfullness"] + eps)) * 100,
        "total_tokens": (abs(combined_matrics[winner]["total_tokens"] - combined_matrics[loser]["total_tokens"]) / (combined_matrics[winner]["total_tokens"] + eps)) * 100,
        "latency_seconds": (abs(combined_matrics[winner]["latency_seconds"] - combined_matrics[loser]["latency_seconds"]) / (combined_matrics[winner]["latency_seconds"] + eps)) * 100,
        "ingestion_time": (abs(combined_matrics[winner]["ingestion_time"] - combined_matrics[loser]["ingestion_time"]) / (combined_matrics[winner]["ingestion_time"] + eps)) * 100
    }

    # 2. Find the metric with the absolute biggest gap (The Fatal Flaw)
    worst_loser_metrics = max(loser_gaps, key=lambda k: float(loser_gaps[k]))
    worst_percentage_gap = loser_gaps[worst_loser_metrics]

    fatal_flaw_insight = ""

    if worst_loser_metrics == "context_recall":
        fatal_flaw_insight = f"FATAL FLAW (CRITICAL DATA BLINDNESS): {loser} completely failed because it misses {worst_percentage_gap:.1f}% of the required information compared to {winner}, rendering it blind to your PDF context."
        calculated_insights["retrieval_insights"].append(fatal_flaw_insight)

    elif worst_loser_metrics == "faithfullness":
        fatal_flaw_insight = f"CRITICAL SAFETY HAZARD: {loser} failed due to a structural trust breakdown. Its faithfulness dropped by {worst_percentage_gap:.1f}% compared to the winner, meaning it frequently invents fake facts (hallucinates)."
        calculated_insights["generation_insights"].append(fatal_flaw_insight)

    elif worst_loser_metrics == "latency_seconds":
        fatal_flaw_insight = f"FATAL FLAW (SYSTEM BOTTLENECK): {loser} failed due to severe processing lag. It increases user wait times by a massive {worst_percentage_gap:.1f}% compared to {winner}, making it completely unviable for production."
        calculated_insights["efficiency_insights"].append(fatal_flaw_insight)
    
    elif worst_loser_metrics == "total_tokens":
        fatal_flaw_insight = f"FATAL FLAW (MONEY PIT): {loser} failed due to extreme cost inefficiency. It consumes {worst_percentage_gap:.1f}% more tokens per query than {winner}, which will exponentially inflate your API bills."
        calculated_insights["efficiency_insights"].append(fatal_flaw_insight)
    
    elif worst_loser_metrics == "context_precision":
        fatal_flaw_insight = f"FATAL NOISE: {loser} failed because its context precision is {worst_percentage_gap:.1f}% worse than the winner, meaning it floods the LLM prompt window with massive walls of irrelevant text fluff."
        calculated_insights["retrieval_insights"].append(fatal_flaw_insight)

    elif worst_loser_metrics == "MRR":
        fatal_flaw_insight = f"POOR RANKING ACCURACY: {loser} failed because its search ranking capability is {worst_percentage_gap:.1f}% worse. The absolute best text answers are buried deeply down its retrieval list."
        calculated_insights["retrieval_insights"].append(fatal_flaw_insight)
    
    elif worst_loser_metrics == "relevancy_scores":
        fatal_flaw_insight = f"RELEVANCY COLLAPSE: {loser} failed because its final answers are {worst_percentage_gap:.1f}% less focused on the user prompt, frequently drifting into completely unrelated topics."
        calculated_insights["retrieval_insights"].append(fatal_flaw_insight)

    elif worst_loser_metrics == "ingestion_time":
        fatal_flaw_insight = f"PROCESSING BOTTLENECK: {loser} failed during data preparation, taking {worst_percentage_gap:.1f}% longer just to parse, chunk, and embed the raw files compared to {winner}."
        calculated_insights["efficiency_insights"].append(fatal_flaw_insight)

    elif worst_loser_metrics == "efficiency_score":
        fatal_flaw_insight = f"POOR RETURN ON INVESTMENT: {loser} failed because its overall operational value profile is {worst_percentage_gap:.1f}% worse than {winner} when balancing quality against operational costs."
        calculated_insights["efficiency_insights"].append(fatal_flaw_insight)

    llm_payload["winner_vs_runnerup_insights"] = calculated_insights

    logger.debug(f"Analyse Matrics Completed LLM Json Payload Ready!!")
    save_state(filename= "llm_json_payload", data = llm_payload)
    return {"llm_json_payload" : llm_payload, "top_scorers" : top_scorers, "combined_matrics" : combined_matrics}
