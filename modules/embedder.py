import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# We'll use a small, fast, very popular model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def load_embedder() -> SentenceTransformer:
    """
    Lazy-load the sentence transformer model only once.
    """
    global _model
    if _model is not None:
        return _model

    print(f"🚀 Loading embedding model: {MODEL_NAME} (first time might be slow)...")
    _model = SentenceTransformer(MODEL_NAME)
    print("✅ Embedding model loaded")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Given a list of strings, return a 2D numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")  # 384 is dim of MiniLM-L6-v2

    model = load_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


if __name__ == "__main__":
    # Tiny test to make sure embeddings work
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Cats are cute animals.",
    ]
    print("🔧 Testing embedder with 2 sample sentences...")
    vecs = embed_texts(sample_texts)
    print(f"Embeddings shape: {vecs.shape}")
    print("First vector (first 5 dims):", vecs[0][:5])
