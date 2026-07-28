from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
import os
import pdfplumber
import logging
logger = logging.getLogger("app") 

def text_loader(state:AgentState):
    # if os.path.exists("src/data/text_loader.json"):
    #     data = load_state("text_loader")
    #     return {"text" : data}
    file_path = state["file_path"]
    logger.info("TEXT LOADER STARTED")
    with pdfplumber.open(file_path) as pdf:
        pdf_content = []
        for page in pdf.pages:
            pages = page.extract_text()
            pdf_content.append(pages)
        full_content = ''.join(pdf_content)
    logger.debug(f"Text Loader Completed")
    save_state(filename= "text_loader", data = full_content)
    return {"text" : full_content}
