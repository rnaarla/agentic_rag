# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from api import app
from models import initialize_models
from utils import download_datasets, load_and_chunk_documents

@pytest.fixture(scope="session")
def test_client():
    # Set up the TestClient for the FastAPI app
    client = TestClient(app)
    yield client

@pytest.fixture(scope="session")
async def setup_data():
    # Setup datasets and models for testing
    await download_datasets()
    doc_splits = await load_and_chunk_documents('./docs')
    embeddings, reranker, llm = initialize_models()
    return {
        "doc_splits": doc_splits,
        "embeddings": embeddings,
        "reranker": reranker,
        "llm": llm,
    }