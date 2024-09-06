# tests/test_models.py
import pytest
from unittest.mock import patch
from models import initialize_models

@patch('models.initialize_models')
def test_initialize_models(mock_initialize):
    # Mock the initialization to return mock objects
    mock_initialize.return_value = ("mock_embeddings", "mock_reranker", "mock_llm")
    
    embeddings, reranker, llm = initialize_models()
    
    assert embeddings == "mock_embeddings", "Embeddings should be mocked"
    assert reranker == "mock_reranker", "Reranker should be mocked"
    assert llm == "mock_llm", "LLM should be mocked"