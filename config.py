import os

# Configuration Constants
DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/docugami/KG-RAG-datasets/main/nih-clinical-trial-protocols/download.csv",
    "https://raw.githubusercontent.com/docugami/KG-RAG-datasets/main/nih-clinical-trial-protocols/download.py"
]
LOCAL_EMBEDDINGS_URL = os.getenv("LOCAL_EMBEDDINGS_URL", "http://<REPLACE_WITH_LOCAL_MACHINE_IP>:8000/v1")
LLM_API_URL = os.getenv("LLM_API_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("API_KEY", "<REPLACE_WITH_GENERATED_API_KEY>")
DOCS_DIRECTORY = './docs'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100