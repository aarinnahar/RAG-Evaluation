from src.orchestration.agent_state import AgentState
from src.prompt_templates.llm_insights_generation import LLM_INSIGHTS_GENERATION_PROMPT
from src.models.schemas.llm_insight_generation_schema import EvaluationReport
from src.utils.save_state import save_state, load_state
from langchain_core.prompts import PromptTemplate
from src.llm.llm_client import get_llm
import logging
logger = logging.getLogger("app") 
import os
import json

def llm_insights_generator(state : AgentState):
    # if os.path.exists("src/utils/data/llm_insights_generator.json"):
    #     logger.info("Skipping big_story_section_writer")
    #     with open("src/utils/data/llm_insights_generator.json", encoding="utf-8") as f:
    #         return {"llm_insights" :json.load(f)}
    llm_json_payload = {**state['llm_json_payload']}
    top_scorers = state['top_scorers']
    winner = top_scorers[0][0]
    runner_up = top_scorers[1][0]
    loser = top_scorers[3][0]
    splitters = {"recursive_text_splitter_chunks" : "Recursive Character Text Splitter", 
                 "character_text_splitter_chunks" : "Character Text Splitter",
                 "semantic_chunks" : "Semantic Text Splitter",
                  "token_text_splitter_chunks" : "Token Text Splitter" }  
    calculated_insights = llm_json_payload["winner_vs_runnerup_insights"] 
    llm_insights = {}

    prompt = PromptTemplate(template= LLM_INSIGHTS_GENERATION_PROMPT, input_variables= ['winner_name', 
                                                                                        "calculated_insights",
                                                                                        'runner_up_name',
                                                                                        "loser_name"])
    llm = get_llm()
    structured_llm = llm.with_structured_output(schema=EvaluationReport, include_raw= True)

    main_prompt = prompt.format(winner_name= splitters[winner], 
                                runner_up_name = splitters[runner_up],
                                loser_name = splitters[loser],
                                calculated_insights=calculated_insights)
    
    message = main_prompt 
    for trial in range(3):
        logger.info(f'Trial run {trial + 1} in llm_insights_generator')
        response = structured_llm.invoke(message)
        if response['parsing_error']:
            logger.warning(f"Parsing error on llm_insights_generator trial {trial + 1}: {response['parsing_error']}")
            # Append the error TO THE MAIN PROMPT so it remembers the rules
            message = main_prompt + f"\n\nSYSTEM WARNING - FIX PREVIOUS ERROR: {response['parsing_error']}"
        else:
            llm_insights = {"retrieval_card" : {"headline" : response['parsed'].retrieval_card.headline,
                                        "body" : response['parsed'].retrieval_card.body,
                                        "hidden_story" : response['parsed'].retrieval_card.hidden_story},
                    "generation_card" :{"headline" : response['parsed'].generation_card.headline,
                                        "body" : response['parsed'].generation_card.body,
                                        "hidden_story" : response['parsed'].generation_card.hidden_story},
                    "efficiency_card" :{"headline" : response['parsed'].efficiency_card.headline,
                                        "body" : response['parsed'].efficiency_card.body,
                                        "hidden_story" : response['parsed'].efficiency_card.hidden_story},
                    "recommendation_summary_text" : response['parsed'].recommendation_summary_text}
            
            break
    else:
        # This 'else' triggers ONLY if the loop finishes without a 'break' (meaning it failed 3 times)
        logger.error(f"Failed to get a valid score for llm_insights_generator after 3 attempts. Assigning 0.")

    logger.debug(f"llm_insights_generator Completed")
    save_state(filename= "llm_insights_generator", data = llm_insights)
    return {"llm_insights" : llm_insights}










