from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Any, Dict
import asyncio
from utils import download_datasets, load_and_chunk_documents
from models import initialize_models
from retrievers import create_hybrid_retriever
from workflow import setup_langgraph_workflow, run_workflow
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define request and response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500, description="The main question to be processed")

    @validator('question')
    def validate_question(cls, value):
        if not value.endswith("?"):
            raise ValueError("Question must end with a question mark.")
        return value

class AnswerResponse(BaseModel):
    sub_questions: List[str]
    answer: str

class SubQuery(BaseModel):
    questions: List[str] = Field(description="The list of sub-questions")

def generate_sub_queries(llm: ChatOpenAI, question: str) -> List[str]:
    """
    Generate sub-queries from the main question using structured generation.

    Args:
        llm (ChatOpenAI): The LLM model instance.
        question (str): The main question.

    Returns:
        List[str]: A list of generated sub-queries.
    """
    try:
        sub_question_generator = llm.with_structured_output(SubQuery)
        result = sub_question_generator.invoke({"input": question})
        sub_queries = result.questions
        logging.info(f"Generated sub-queries for the question: {question}")
        return sub_queries
    except Exception as e:
        logging.error(f"Failed to generate sub-queries: {e}")
        return []

def create_rag_chain(llm: ChatOpenAI) -> Any:
    """
    Create a RAG (Retrieval-Augmented Generation) chain.

    Args:
        llm (ChatOpenAI): The LLM model instance.

    Returns:
        Any: A configured RAG chain.
    """
    try:
        # Pull a pre-defined RAG prompt template from the LangChain Hub
        prompt = hub.pull("rlm/rag-prompt")
        # Combine the prompt with the LLM and an output parser
        rag_chain = prompt | llm | StrOutputParser()
        logging.info("RAG chain created successfully.")
        return rag_chain
    except Exception as e:
        logging.error(f"Failed to create RAG chain: {e}")
        raise

async def initialize_app() -> Dict[str, Any]:
    """
    Initializes the datasets and models asynchronously.
    """
    await download_datasets()
    doc_splits = await load_and_chunk_documents('./docs')
    embeddings, reranker, llm = initialize_models()
    hybrid_retriever = create_hybrid_retriever(doc_splits, embeddings)
    return {
        "llm": llm,
        "hybrid_retriever": hybrid_retriever,
    }

# Initialize app data
app_data = asyncio.run(initialize_app())
llm = app_data["llm"]
hybrid_retriever = app_data["hybrid_retriever"]

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to handle questions and generate answers using the RAG workflow.
    This endpoint supports parallelism and concurrency to handle multiple requests.

    Args:
        request (QuestionRequest): The incoming request containing the question.
        background_tasks (BackgroundTasks): Allows background processing of tasks.

    Returns:
        AnswerResponse: The response containing sub-queries and the generated answer.
    """
    try:
        question = request.question
        sub_queries = generate_sub_queries(llm, question)

        if not sub_queries:
            raise HTTPException(status_code=400, detail="Failed to generate sub-queries")

        rag_chain = create_rag_chain(llm)

        # Define a background task to run the workflow concurrently
        def run_rag_workflow(inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                app_workflow = setup_langgraph_workflow(llm, sub_queries, hybrid_retriever, rag_chain)
                result = run_workflow(app_workflow, inputs)
                return result
            except Exception as e:
                logging.error(f"Error in workflow execution: {e}")
                return {"generation": "Failed to generate answer"}

        inputs = {"question": question}
        result = await asyncio.to_thread(run_rag_workflow, inputs)  # Run in a separate thread

        # Assume the final answer is stored in the result after workflow execution
        final_answer = result.get("generation", "No answer generated")
        
        return AnswerResponse(sub_questions=sub_queries, answer=final_answer)

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error processing the question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Default root endpoint
@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Multi-Agent RAG API"}