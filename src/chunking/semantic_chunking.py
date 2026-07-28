from src.config.settings import Settings
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import re
from langchain_huggingface import HuggingFaceEmbeddings
import html
from bs4 import BeautifulSoup
import logging
logger = logging.getLogger("app") 


def embedding_model():
    model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    return model


def robust_clean(raw_text):
    # 1. Convert entities like &lt; to <
    unescaped_text = html.unescape(raw_text)
    
    # 2. Strip actual HTML tags (like <html>, <body>)
    soup = BeautifulSoup(unescaped_text, "html.parser")
    clean_text = soup.get_text()
    
    # 3. Remove "Literal" tags left behind (like <p> or </body>)
    # This regex looks for anything between < and >
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    
    # 4. Final Polish: Fix whitespaces and newlines
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


# Load these ONCE at the top level, not inside the function
nlp = spacy.load("en_core_web_sm", disable=['ner'])
# Assume 'model' is initialized globally
# model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def semantic_chunking_pro(texts, window_size=3, percentile=10):
    model = embedding_model()
    text = robust_clean(texts)
    # 1. Split sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
    
    if len(sentences) < window_size:
        return [" ".join(sentences)]

    # 2. Batch Embedding (MUCH faster)
    vectors = np.array(model.embed_documents(sentences))

    # 3. Sliding Window Means (Vectorized)
    windows = sliding_window_view(vectors, window_shape=(window_size,), axis=0)
    mean_vectors = windows.mean(axis=1)

    # 4. Calculate PAIRWISE Similarity (Neighbor vs Neighbor)
    # We compare mean_vector[i] with mean_vector[i+1]
    similarities = []
    for i in range(len(mean_vectors) - 1):
        v1 = mean_vectors[i].reshape(1, -1)
        v2 = mean_vectors[i+1].reshape(1, -1)
        sim = cosine_similarity(v1, v2)[0][0]
        similarities.append(sim)

    # 5. Thresholding
    threshold = np.percentile(similarities, percentile)
    
    # 6. Find Breakpoints
    # We add 1 because the similarity is between index i and i+1
    # We also add (window_size // 2) to align the cut with the window center
    offset = window_size // 2
    breakpoints = [i + offset for i, s in enumerate(similarities) if s <= threshold]

    # 7. Slicing
    chunks = []
    start_idx = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start_idx:bp]))
        start_idx = bp
    
    chunks.append(" ".join(sentences[start_idx:]))
    
    return chunks
