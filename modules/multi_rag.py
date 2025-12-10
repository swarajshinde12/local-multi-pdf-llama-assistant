import os
import sys
from typing import List, Tuple

import numpy as np

# 🔧 Make sure project root (C:\local_ai) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.multi_pdf_loader import load_multiple_pdfs
from modules.text_splitter import split_text_into_chunks
from modules.embedder import embed_texts
from modules.vector_store import VectorStore
from modules.local_llm import generate_answer


def build_vector_store_from_folder(
    folder_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Tuple[VectorStore, List[str]]:
    """
    1. Load ALL PDFs in folder
    2. Merge text from all PDFs
    3. Split into chunks
    4. Embed chunks
    5. Build FAISS vector store
    Returns: (store, chunks_list)
    """
    print(f"📁 Building multi-PDF vector store from folder: {folder_path}")

    pdf_texts = load_multiple_pdfs(folder_path)  # {filename: text}

    if not pdf_texts:
        raise ValueError("No valid PDFs with extractable text found.")

    # Merge all texts into one big list of chunks
    all_chunks: List[str] = []

    for name, text in pdf_texts.items():
        print(f"✂️ Splitting PDF '{name}' into chunks...")
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"   ➜ {len(chunks)} chunks from {name}")
        all_chunks.extend(chunks)

    print(f"✅ Total chunks from all PDFs: {len(all_chunks)}")

    print("🧮 Embedding all chunks...")
    embeddings = embed_texts(all_chunks)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    store = VectorStore(dim=dim)

    print("📦 Adding embeddings to vector store...")
    store.add_embeddings(embeddings, all_chunks)
    print("✅ Multi-PDF vector store ready")

    return store, all_chunks


def answer_question_multi_pdf(
    store: VectorStore,
    question: str,
    top_k: int = 5,
) -> str:
    """
    Same as single-PDF RAG, but using the multi-PDF vector store.
    """
    from modules.embedder import embed_texts

    print(f"❓ User question: {question}")

    # 1) Embed question
    q_emb = embed_texts([question])
    q_vec = q_emb[0]

    # 2) Search in FAISS
    results = store.search(q_vec, top_k=top_k)

    if not results:
        print("⚠️ No similar chunks found.")
        context_text = ""
    else:
        print(f"📚 Top {len(results)} chunks retrieved from ALL PDFs:")
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
            "Explain that the PDFs did not contain enough information."
        )

    print("🤖 Sending prompt to local LLM...")
    answer = generate_answer(prompt, max_new_tokens=256)
    print("✅ Got answer from LLM")
    return answer


if __name__ == "__main__":
    # Small end-to-end test
    folder = r"C:\local_ai\data"

    print("🚀 Building multi-PDF store from folder...")
    store, chunks = build_vector_store_from_folder(folder)

    test_question = "What is the main topic across these documents?"
    ans = answer_question_multi_pdf(store, test_question, top_k=5)

    print("\n🧠 Final answer (multi-PDF):\n")
    print(ans)
