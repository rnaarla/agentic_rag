# tests/test_api.py
import pytest

def test_api_root(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Multi-Agent RAG API"}

@pytest.mark.asyncio
async def test_api_ask_question(test_client):
    response = test_client.post("/ask", json={"question": "What is the relationship between X and Y?"})
    assert response.status_code == 200
    assert "sub_questions" in response.json()
    assert "answer" in response.json()