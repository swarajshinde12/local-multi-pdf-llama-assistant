import os
import streamlit as st

from modules.pipeline import build_vector_store_from_pdf, answer_question_with_rag

PDF_PATH = r"C:\local_ai\data\sample.pdf"

st.set_page_config(page_title="Local PDF Chat", page_icon="📄", layout="centered")

st.title("📄 Local PDF Chat (Offline RAG)")
st.caption("Ask questions about sample.pdf using a local LLM (no API keys).")

# -------------------
# Build vector store once per session
# -------------------
if "store" not in st.session_state:
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF not found at {PDF_PATH}. Put a file named sample.pdf there.")
    else:
        with st.spinner("Indexing PDF into vector store..."):
            store, chunks = build_vector_store_from_pdf(PDF_PATH)
        st.session_state.store = store
        st.success("PDF indexed successfully ✅")

# -------------------
# Chat UI
# -------------------
question = st.text_input("❓ Ask a question about this PDF:")

if st.button("Ask") and question.strip():
    if "store" not in st.session_state:
        st.error("Vector store not ready. Check PDF path.")
    else:
        with st.spinner("Thinking with PDF + local LLM..."):
            answer = answer_question_with_rag(st.session_state.store, question, top_k=5)

        st.subheader("🧠 Answer")
        st.write(answer)
