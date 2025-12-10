import os
import streamlit as st

from modules.multi_rag_gguf import (
    build_vector_store_from_folder_gguf,
    answer_question_multi_pdf_gguf,
)

DATA_FOLDER = r"C:\local_ai\data"

st.set_page_config(
    page_title="Local Multi-PDF Chat (GGUF LLaMA)",
    page_icon="🦙",
    layout="centered",
)

st.title("🦙 Local Multi-PDF Chat (GGUF LLaMA 3.1 8B)")
st.caption("Ask questions across ALL PDFs in /data using a fully local LLaMA model (no API keys).")

# -------------------
# Build vector store once
# -------------------
if "gguf_store" not in st.session_state:
    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder not found: {DATA_FOLDER}")
    else:
        with st.spinner("Indexing ALL PDFs in /data into a vector store..."):
            store, chunks = build_vector_store_from_folder_gguf(DATA_FOLDER)
        st.session_state.gguf_store = store
        st.success("Multi-PDF index ready ✅")

# -------------------
# Simple chat UI
# -------------------
question = st.text_input("❓ Ask a question about your PDFs:")

if st.button("Ask with LLaMA") and question.strip():
    if "gguf_store" not in st.session_state:
        st.error("Vector store not ready. Check data folder and restart app.")
    else:
        with st.spinner("Thinking with PDFs + LLaMA 3.1 8B..."):
            answer = answer_question_multi_pdf_gguf(st.session_state.gguf_store, question, top_k=5)

        st.subheader("🧠 Answer")
        st.write(answer)
