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
- [Docker Setup](#docker-setup)
- [Prometheus and Grafana Integration](#prometheus-and-grafana-integration)
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

## Project Directory Structure
project-root/
│
├── .devcontainer/
│   ├── devcontainer.json        # Configuration for GitHub Codespaces and VS Code Remote - Containers
│   └── Dockerfile               # Custom Dockerfile for the dev container
│
├── tests/                       # Directory containing all test files
│   ├── __init__.py              # Marks the tests directory as a package
│   ├── conftest.py              # Common fixtures for tests
│   ├── test_api.py              # Tests for the API endpoints
│   ├── test_models.py           # Unit tests for the models
│   ├── test_utils.py            # Unit tests for utility functions
│   └── test_workflow.py         # Integration tests for the workflow
│
├── .dockerignore                # Excludes unnecessary files from Docker builds
├── .env                         # Environment variables for sensitive data
├── .gitignore                   # Excludes unnecessary files from Git commits
├── api.py                       # Main FastAPI application file
├── config.py                    # Configuration settings and constants
├── docker-compose.yml           # Docker Compose configuration for multi-service setups
├── gunicorn_conf.py             # Configuration for running Gunicorn with Uvicorn workers
├── main.py                      # Entry point for running the multi-agent RAG workflow
├── models.py                    # Functions for initializing models and related tasks
├── nginx.conf                   # NGINX configuration for SSL/TLS and reverse proxy setup
├── prometheus.yml               # Prometheus configuration for monitoring
├── README.md                    # Detailed instructions and documentation for the project
├── requirements.txt             # List of Python dependencies
├── retrievers.py                # Setup and configuration of hybrid retrieval models
├── utils.py                     # Utility functions for various tasks
└── workflow.py                  # Definition of the LangGraph-based multi-agent workflow


## Testing
- **Unit Tests:** Write unit tests for individual functions using pytest.
- **Integration Tests:** Test the workflow end-to-end by simulating realistic data inputs and expected outputs.
- **Run Tests:**
	```bash
	pytest tests/

## Docker Setup
- **Building and Running with Docker**
1. **Build the Docker Image**
	```Bash
	docker build -t my-rag-app .
	Run the Docker Container
	docker run --env-file .env -p 8000:8000 my-rag-app

**Using Docker Compose**
If you prefer using Docker Compose, run:
	```Bash
	docker-compose up --build


- **Docker Compose Configuration**
Ensure your docker-compose.yml includes the application and optionally NGINX for handling SSL/TLS:
	```yaml
	version: '3.8'

	services:
	app:
		build:
		context: .
		dockerfile: Dockerfile
		environment:
		- LOCAL_EMBEDDINGS_URL=${LOCAL_EMBEDDINGS_URL}
		- LLM_API_URL=${LLM_API_URL}
		- API_KEY=${API_KEY}
		volumes:
		- .:/app
		expose:
		- "8000"

	nginx:
		image: nginx:latest
		ports:
		- "80:80"
		- "443:443"
		volumes:
		- ./nginx.conf:/etc/nginx/nginx.conf
		depends_on:
		- app

## Prometheus and Grafana Integration**
- **Monitoring with Prometheus and Grafana**
1. **Set Up Prometheus: Create a prometheus.yml configuration file:**
	```yaml
	global:
	scrape_interval: 15s

	scrape_configs:
	- job_name: 'api'
		static_configs:
		- targets: ['app:8000']  # Assuming Uvicorn exposes metrics at /metrics

2. **Include Prometheus and Grafana in Docker Compose**
	```yaml
	prometheus:
	image: prom/prometheus:latest
	volumes:
		- ./prometheus.yml:/etc/prometheus/prometheus.yml
	ports:
		- "9090:9090"

	grafana:
	image: grafana/grafana:latest
	ports:
		- "3000:3000"
	environment:
		- GF_SECURITY_ADMIN_PASSWORD=admin  # Set your Grafana admin password
	depends_on:
		- prometheus

Access Grafana
Visit http://localhost:3000 and log in with the default username (admin) and password (admin). Add Prometheus as a data source and create dashboards to monitor your application's metrics.