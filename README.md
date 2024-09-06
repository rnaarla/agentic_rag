# Multi-Agent Retrieval-Augmented Generation (RAG) Workflow

This project implements a multi-agent Retrieval-Augmented Generation (RAG) workflow using Python. It leverages advanced data structures, parallel processing, and machine learning models to efficiently handle and process large datasets of clinical trial documents. The project uses a combination of retrieval models (BM25, FAISS), embedding models, reranking models, and LLMs (Large Language Models) to provide a robust question-answering system.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Code Structure](#code-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

The workflow is designed to process documents by:
1. Downloading datasets asynchronously.
2. Loading and chunking documents for processing.
3. Initializing models for embeddings, reranking, and large language model (LLM) interaction.
4. Creating a hybrid retriever combining BM25 and FAISS.
5. Setting up a LangGraph-based workflow for multi-agent decision making.
6. Executing a multi-step RAG pipeline with error handling and logging for robustness.

## Features

- **Asynchronous Processing:** Improves the speed of dataset downloads and document processing.
- **Hybrid Retrieval:** Combines keyword-based (BM25) and semantic (FAISS) retrieval for improved accuracy.
- **Modular Architecture:** Organized into separate modules for easy maintenance and extension.
- **Error Handling and Logging:** Comprehensive error handling and logging for monitoring and debugging.
- **Flexible Configuration:** Uses environment variables for sensitive data like API keys and URLs.
- **Testing Support:** Recommendations for unit and integration tests to ensure code reliability.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/multi-agent-rag-workflow.git
   cd multi-agent-rag-workflow
2. **Create a Virtual Environment:**
	```bash
	python3 -m venv env
	source env/bin/activate  # On Windows use `env\Scripts\activate`
3. **Install Dependencies:**
	``` bash
	pip install -r requirements.txt
4. **Set Up Environment Variables:** Create a .env file in the root directory and add 
   your configuration:
   ```bash
	LOCAL_EMBEDDINGS_URL="http://<REPLACE_WITH_LOCAL_MACHINE_IP>:8000/v1"
	LLM_API_URL="https://integrate.api.nvidia.com/v1"
	API_KEY="<REPLACE_WITH_GENERATED_API_KEY>"
## Usage
1. **Run the Main Workflow:** To start the multi-agent RAG workflow, execute the main script:
	```bash
	python main.py
2. **Customize Configurations:**
- **Update config.py** for custom settings like chunk size and overlap.
- Adjust URLs and API keys in your **.env** file.
- **Monitor Logs:** Logs are written to the console to help track progress and debug issues. Adjust logging levels as needed in utils.py.

## Configuration
- **Environment Variables:** Use .env or system environment variables to manage sensitive data securely.
- **Logging:** Configurable logging settings are available in utils.py.
- **Chunking and Splitting:** Adjust the chunk size and overlap in config.py as per the dataset's requirements.

## Code Structure
- **config.py:** Contains configuration constants and environment variable setups.
- **utils.py:** Utility functions for downloading datasets and processing documents.
- **models.py:** Functions for initializing embedding, reranking, and LLM models.
- **retrievers.py:** Setup and configuration of hybrid retrieval models.
- **workflow.py:** Defines the LangGraph-based multi-agent workflow.
- **main.py:** Orchestrates the entire workflow execution.

## Testing
- **Unit Tests:** Write unit tests for individual functions using pytest.
- **Integration Tests:** Test the workflow end-to-end by simulating realistic data inputs and expected outputs.
- **Run Tests:**
	```bash
	pytest tests/