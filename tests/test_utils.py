# tests/test_utils.py
import pytest
from utils import download_datasets, load_and_chunk_documents

@pytest.mark.asyncio
async def test_download_datasets():
    # Test if datasets download without errors
    try:
        await download_datasets()
    except Exception as e:
        pytest.fail(f"Dataset download failed: {e}")

@pytest.mark.asyncio
async def test_load_and_chunk_documents():
    # Assuming there are sample PDFs in the './docs' directory for testing
    try:
        doc_splits = await load_and_chunk_documents('./docs')
        assert isinstance(doc_splits, list), "Document splits should be a list"
        assert len(doc_splits) > 0, "Document splits should not be empty"
    except Exception as e:
        pytest.fail(f"Loading and chunking documents failed: {e}")
