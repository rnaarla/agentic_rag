from typing import List
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import logging

def create_hybrid_retriever(doc_splits: List, embeddings: NVIDIAEmbeddings) -> EnsembleRetriever:
    """
    Create a hybrid retriever combining BM25 and FAISS retrievers.

    Args:
        doc_splits (List): List of document splits.
        embeddings (NVIDIAEmbeddings): The embeddings model.

    Returns:
        EnsembleRetriever: A hybrid retriever instance.
    """
    try:
        bm25_retriever = BM25Retriever.from_documents(doc_splits)
        faiss_vectorstore = FAISS.from_documents(doc_splits, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        logging.info("Hybrid retriever created successfully.")
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.7, 0.3])
    except Exception as e:
        logging.error(f"Failed to create hybrid retriever: {e}")
        raise