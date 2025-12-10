import os
import sys
from typing import List, Tuple

import numpy as np

# 🔧 Ensure project root (C:\local_ai) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.multi_pdf_loader import load_multiple_pdfs
from modules.text_splitter import split_text_into_chunks
from modules.embedder import embed_texts
from modules.vector_store import VectorStore
from modules.local_llm_gguf import generate_answer as gguf_generate_answer


def build_vector_store_from_folder_gguf(
    folder_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Tuple[VectorStore, List[str]]:
    """
    Same as build_vector_store_from_folder, but named for clarity.
    1. Load ALL PDFs in folder
    2. Split into chunks
    3. Embed
    4. Build FAISS index
    """
    print(f"📁 [GGUF] Building multi-PDF vector store from folder: {folder_path}")

    pdf_texts = load_multiple_pdfs(folder_path)

    if not pdf_texts:
        raise ValueError("No valid PDFs with extractable text found.")

    all_chunks: List[str] = []

    for name, text in pdf_texts.items():
        print(f"✂️ [GGUF] Splitting PDF '{name}' into chunks...")
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"   ➜ {len(chunks)} chunks from {name}")
        all_chunks.extend(chunks)

    print(f"✅ [GGUF] Total chunks from all PDFs: {len(all_chunks)}")

    print("🧮 [GGUF] Embedding all chunks...")
    embeddings = embed_texts(all_chunks)
    print(f"✅ [GGUF] Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    store = VectorStore(dim=dim)

    print("📦 [GGUF] Adding embeddings to vector store...")
    store.add_embeddings(embeddings, all_chunks)
    print("✅ [GGUF] Multi-PDF vector store ready")

    return store, all_chunks


def answer_question_multi_pdf_gguf(
    store: VectorStore,
    question: str,
    top_k: int = 5,
) -> str:
    """
    Multi-PDF RAG answerer, but uses local GGUF LLaMA instead of Flan-T5.
    """
    from modules.embedder import embed_texts

    print(f"❓ [GGUF] User question: {question}")

    # 1) Embed question
    q_emb = embed_texts([question])
    q_vec = q_emb[0]

    # 2) Search in FAISS
    results = store.search(q_vec, top_k=top_k)

    if not results:
        print("⚠️ [GGUF] No similar chunks found.")
        context_text = ""
    else:
        print(f"📚 [GGUF] Top {len(results)} chunks retrieved from ALL PDFs:")
        context_parts = []
        for i, (chunk, score) in enumerate(results, start=1):
            print(f"  - Chunk {i}, score={score:.3f}, length={len(chunk)}")
            context_parts.append(f"[Chunk {i}, score={score:.3f}]\n{chunk}\n")
        context_text = "\n".join(context_parts)

    if context_text.strip():
        prompt = (
            "You are a helpful assistant. Use ONLY the following PDF context (from multiple documents) "
            "to answer the question.\n\n"
            "PDF CONTEXT:\n"
            f"{context_text}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Answer in a clear, concise way."
        )
    else:
        prompt = (
            "You are a helpful assistant. The user asked a question, but the PDFs did not yield relevant context.\n\n"
            f"QUESTION:\n{question}\n\n"
            "Explain clearly that the PDFs did not contain enough information."
        )

    print("🤖 [GGUF] Sending prompt to local GGUF LLaMA...")
    answer = gguf_generate_answer(prompt, max_tokens=256)
    print("✅ [GGUF] Got answer from GGUF model")
    return answer


if __name__ == "__main__":
    folder = r"C:\local_ai\data"

    print("🚀 [GGUF] Building multi-PDF store from folder...")
    store, chunks = build_vector_store_from_folder_gguf(folder)

    test_question = "What is the main topic across these documents?"
    ans = answer_question_multi_pdf_gguf(store, test_question, top_k=5)

    print("\n🧠 [GGUF] Final answer (multi-PDF):\n")
    print(ans)
