import asyncio
import aiohttp
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from config import DOWNLOAD_URLS, DOCS_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def download_file(url: str) -> None:
    """
    Download a file asynchronously from a given URL.

    Args:
        url (str): The URL to download the file from.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                filename = url.split("/")[-1]
                with open(filename, "wb") as f:
                    f.write(await response.read())
        logging.info(f"Downloaded {filename} successfully.")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")

async def download_datasets() -> None:
    """
    Download all datasets asynchronously.
    """
    try:
        await asyncio.gather(*[download_file(url) for url in DOWNLOAD_URLS])
    except Exception as e:
        logging.error(f"Error downloading datasets: {e}")

async def load_and_chunk_documents(directory: str) -> List:
    """
    Load and split documents into chunks asynchronously.

    Args:
        directory (str): The directory from which to load the documents.

    Returns:
        List: A list of document chunks.
    """
    try:
        loader = DirectoryLoader(directory, glob="**/*.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logging.info(f"Loaded and chunked documents from {directory}.")
        return text_splitter.split_documents(docs)
    except Exception as e:
        logging.error(f"Failed to load or chunk documents: {e}")
        return []