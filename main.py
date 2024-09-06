import asyncio
from utils import download_datasets, load_and_chunk_documents
from models import initialize_models
from retrievers import create_hybrid_retriever
from workflow import setup_langgraph_workflow, run_workflow
import logging

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

        # Step 5: Generate sub-queries and setup workflow
        question = "How does Get It Right First Time (GIRFT) Urology programme relate to TURBT and URS?"
        sub_queries = generate_sub_queries(llm, question)
        rag_chain = create_rag_chain(llm)

        # Step 6: Setup and run LangGraph workflow
        app = setup_langgraph_workflow(llm, sub_queries, hybrid_retriever, rag_chain)
        inputs = {"question": question}
        run_workflow(app, inputs)

    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")

# Execute the main workflow
if __name__ == "__main__":
    asyncio.run(main())