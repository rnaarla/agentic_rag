import asyncio
from utils import download_datasets, load_and_chunk_documents
from models import initialize_models
from retrievers import create_hybrid_retriever
from workflow import setup_langgraph_workflow, run_workflow
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from pydantic import BaseModel, Field
from typing import List, Any, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

async def main() -> None:
    """
    Main function to orchestrate the full workflow execution.
    """
    try:
        # Step 1: Download datasets
        await download_datasets()

        # Step 2: Load and process documents concurrently
        doc_splits = await load_and_chunk_documents('./docs')

        # Step 3: Initialize models
        embeddings, reranker, llm = initialize_models()

        # Step 4: Create hybrid retriever
        hybrid_retriever = create_hybrid_retriever(doc_splits, embeddings)

        # Define a question
        question = "How does Get It Right First Time (GIRFT) Urology programme relate to TURBT and URS?"

        # Step 5: Generate sub-queries using the LLM
        sub_queries = generate_sub_queries(llm, question)
        if not sub_queries:
            logging.error("No sub-queries generated, aborting workflow.")
            return

        # Step 6: Create the RAG chain using the LLM
        rag_chain = create_rag_chain(llm)

        # Step 7: Setup and run LangGraph workflow
        app = setup_langgraph_workflow(llm, sub_queries, hybrid_retriever, rag_chain)
        inputs = {"question": question}
        run_workflow(app, inputs)

    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")

# Execute the main workflow
if __name__ == "__main__":
    asyncio.run(main())