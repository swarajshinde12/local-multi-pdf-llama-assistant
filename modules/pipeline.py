import os
from typing import List, Tuple
import numpy as np

from .pdf_loader import load_pdf_text
from .text_splitter import split_text_into_chunks
from .embedder import embed_texts
from .vector_store import VectorStore
from .local_llm import generate_answer



def build_vector_store_from_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Tuple[VectorStore, List[str]]:
    """
    1. Load PDF text
    2. Split into chunks
    3. Embed chunks
    4. Build and return a FAISS vector store
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"📄 Loading PDF text from: {pdf_path}")
    full_text = load_pdf_text(pdf_path)

    if not full_text.strip():
        raise ValueError("No text extracted from PDF. It might be scanned/image-only or corrupted.")

    print("✂️ Splitting text into chunks...")
    chunks = split_text_into_chunks(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"✅ Total chunks: {len(chunks)}")

    print("🧮 Embedding chunks...")
    embeddings = embed_texts(chunks)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    store = VectorStore(dim=dim)

    print("📦 Adding embeddings to vector store...")
    store.add_embeddings(embeddings, chunks)
    print("✅ Vector store ready")

    return store, chunks


def answer_question_with_rag(
    store: VectorStore,
    question: str,
    top_k: int = 5,
) -> str:
    """
    1. Embed the question
    2. Retrieve top-k similar chunks
    3. Build a context prompt
    4. Ask the local LLM to answer using that context
    """
    print(f"❓ User question: {question}")

    # 1) Embed question
    from .embedder import embed_texts

    q_emb = embed_texts([question])  # shape (1, dim)
    q_vec = q_emb[0]

    # 2) Retrieve top-k chunks from FAISS
    results = store.search(q_vec, top_k=top_k)

    if not results:
        print("⚠️ No results from vector store")
        context_text = ""
    else:
        print(f"📚 Top {len(results)} chunks retrieved:")
        context_parts = []
        for i, (chunk, score) in enumerate(results, start=1):
            print(f"  - Chunk {i}, score={score:.3f}, length={len(chunk)}")
            context_parts.append(f"[Chunk {i}, score={score:.3f}]\n{chunk}\n")

        context_text = "\n".join(context_parts)

    # 3) Build prompt for LLM
    if context_text.strip():
        prompt = (
            "You are a helpful assistant. Use ONLY the following PDF context to answer the question.\n\n"
            "PDF CONTEXT:\n"
            f"{context_text}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Answer in a clear, concise way."
        )
    else:
        prompt = (
            "You are a helpful assistant. The user asked a question, but the PDF context is empty.\n\n"
            f"QUESTION:\n{question}\n\n"
            "Explain that the PDF did not contain relevant information."
        )

    # 4) Generate answer with local LLM
    print("🤖 Sending prompt to local LLM...")
    answer = generate_answer(prompt, max_new_tokens=256)
    print("✅ Got answer from LLM")
    return answer


if __name__ == "__main__":
    # Small end-to-end test on sample.pdf
    pdf_path = r"C:\local_ai\data\sample.pdf"

    if not os.path.exists(pdf_path):
        print("⚠️ sample.pdf not found in C:\\local_ai\\data\\")
        print("Put a PDF there and run again.")
    else:
        print("🚀 Building vector store from sample.pdf...")
        store, chunks = build_vector_store_from_pdf(pdf_path)

        # Try a test question (you can change this)
        test_question = "What is this document mainly about?"
        ans = answer_question_with_rag(store, test_question, top_k=5)

        print("\n🧠 Final answer:\n")
        print(ans)
