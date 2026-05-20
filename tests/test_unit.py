import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
import torch  # Added to check device availability
import faiss


# --- Fixtures ---
@pytest.fixture
def embedding_model():
    # Safely detect if CUDA is initialized and compatible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If cuda is technically found but has driver initialization errors,
    # torch.cuda.is_available() might still return False, keeping your tests safe.
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


@pytest.fixture
def sample_sentences():
    return [
        "The proposed model achieved a 95% accuracy.",
        "This improvement is due to the attention mechanism.",
    ]


# --- Tests ---
def test_sentence_level_embedding(embedding_model, sample_sentences):
    """Ensures embeddings are generated precisely for each sentence."""
    embeddings = embedding_model.encode(sample_sentences)

    assert len(embeddings) == len(sample_sentences)
    assert len(embeddings[0]) == 384
    assert isinstance(embeddings, np.ndarray)


def test_faiss_index_creation():
    """Verifies FAISS index is initialized with the correct dimensions."""
    dimension = 384
    index = faiss.IndexFlatL2(dimension)

    assert index.is_trained is True
    assert index.ntotal == 0
