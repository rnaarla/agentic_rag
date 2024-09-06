from typing import Tuple
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from config import LOCAL_EMBEDDINGS_URL, LLM_API_URL, API_KEY
import logging

def initialize_models() -> Tuple[NVIDIAEmbeddings, NVIDIARerank, ChatOpenAI]:
    """
    Initialize embedding, reranking, and LLM models.

    Returns:
        Tuple: Instances of NVIDIAEmbeddings, NVIDIARerank, and ChatOpenAI.
    """
    try:
        embeddings = NVIDIAEmbeddings(base_url=LOCAL_EMBEDDINGS_URL, model="nvidia/nv-embedqa-e5-v5", truncate="END")
        reranker = NVIDIARerank(base_url=LOCAL_EMBEDDINGS_URL, model="nvidia/nv-rerankqa-mistral-4b-v3", truncate="END")
        llm = ChatOpenAI(base_url=LLM_API_URL, api_key=API_KEY, model="meta/llama-3.1-405b-instruct")
        logging.info("Models initialized successfully.")
        return embeddings, reranker, llm
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        raise