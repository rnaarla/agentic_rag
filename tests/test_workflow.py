# tests/test_workflow.py
import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_root(test_client):
    # Test the root endpoint
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Multi-Agent RAG API"}

@pytest.mark.asyncio
async def test_ask_question(test_client):
    # Test the ask_question endpoint with a realistic question
    response = test_client.post("/ask", json={"question": "How does Get It Right First Time (GIRFT) Urology programme relate to TURBT and URS?"})
    assert response.status_code == 200
    data = response.json()
    assert "sub_questions" in data, "Response should include sub_questions"
    assert isinstance(data["sub_questions"], list), "sub_questions should be a list"
    assert len(data["sub_questions"]) > 0, "sub_questions should not be empty"
    assert "answer" in data, "Response should include an answer"
    assert isinstance(data["answer"], str), "Answer should be a string"