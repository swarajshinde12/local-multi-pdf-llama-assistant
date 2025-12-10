from typing import List, Tuple
import faiss
import numpy as np


class VectorStore:
    """
    Simple FAISS-based vector store for RAG:
    - add_embeddings() to store vectors + texts
    - search() to retrieve top-k similar chunks
    """

    def __init__(self, dim: int):
        self.dim = dim
        # Cosine similarity via inner product on normalized vectors
        self.index = faiss.IndexFlatIP(dim)
        self.text_chunks: List[str] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        """
        embeddings: shape (n, dim)
        chunks: list of strings, same length n
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings and chunks must match")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normalized = embeddings / norms

        self.index.add(normalized.astype("float32"))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        query_embedding: shape (dim,)
        returns: list of (chunk_text, score)
        """
        if self.index.ntotal == 0:
            return []

        # Normalize
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        q = q.astype("float32").reshape(1, -1)

        scores, indices = self.index.search(q, top_k)
        results: List[Tuple[str, float]] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            chunk_text = self.text_chunks[idx]
            results.append((chunk_text, float(score)))

        return results


if __name__ == "__main__":
    # Tiny test of the vector store
    dim = 4
    store = VectorStore(dim=dim)

    # Fake embeddings
    emb = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0.7, 0.7, 0, 0],
        ],
        dtype="float32",
    )
    texts = ["chunk A", "chunk B", "chunk C"]
    store.add_embeddings(emb, texts)

    query = np.array([0.6, 0.8, 0, 0], dtype="float32")
    results = store.search(query, top_k=2)
    print("Results:", results)
