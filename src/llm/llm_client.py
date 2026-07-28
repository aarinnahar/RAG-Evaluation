from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama 
from src.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    model = ChatOllama(
        model= "qwen2.5:7b",
        temperature= 0.0,
        base_url = "https://carey-dissatisfied-disingenuously.ngrok-free.dev"
    )
    return model