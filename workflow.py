from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
import logging

class GraphState(TypedDict):
    question: str
    sub_questions: List[str]
    generation: str
    documents: List[str]

def setup_langgraph_workflow(
    llm: ChatOpenAI, 
    sub_question_generator: Any, 
    hybrid_retriever: EnsembleRetriever, 
    rag_chain: Any
) -> Any:
    """
    Setup the LangGraph workflow for multi-agent RAG.

    Args:
        llm (ChatOpenAI): The LLM model instance.
        sub_question_generator (Any): Sub-query generator instance.
        hybrid_retriever (EnsembleRetriever): Hybrid retriever instance.
        rag_chain (Any): RAG chain instance.

    Returns:
        A compiled workflow graph.
    """
    def decompose(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        sub_queries = sub_question_generator.invoke(question)
        return {"sub_questions": sub_queries.questions, "question": question}

    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        sub_questions = state["sub_questions"]
        documents = []
        for sub_question in sub_questions:
            docs = hybrid_retriever.get_relevant_documents(sub_question)
            documents.extend(docs)
        return {"documents": documents, "question": state["question"]}

    def rerank(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        documents = state["documents"]
        documents = reranker.compress_documents(query=question, documents=documents)
        return {"documents": documents, "question": question}

    def generate(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    workflow = StateGraph(GraphState)
    workflow.add_node("decompose", decompose)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    logging.info("LangGraph workflow setup successfully.")
    return workflow.compile()

def run_workflow(app: Any, inputs: Dict[str, Any]) -> None:
    """
    Execute the workflow with provided inputs.

    Args:
        app (Any): Compiled workflow application.
        inputs (Dict[str, Any]): Input data for the workflow.
    """
    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                logging.info(f"Node '{key}': {value}")
            logging.info("\n---\n")
    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")