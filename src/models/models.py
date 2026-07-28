from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def embedding_model():
    model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    return model

def embd_model_name():
    model = embedding_model()
    return model.model_name