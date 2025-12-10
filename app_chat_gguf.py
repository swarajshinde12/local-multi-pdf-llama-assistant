import os
import streamlit as st

from modules.multi_rag_gguf import (
    build_vector_store_from_folder_gguf,
    answer_question_multi_pdf_gguf,
)

DATA_FOLDER = r"C:\local_ai\data"

st.set_page_config(
    page_title="🦙 Local LLaMA PDF Chat",
    page_icon="🦙",
    layout="wide",
)

# --------------------------
# SIDEBAR – PDFs + controls
# --------------------------
with st.sidebar:
    st.markdown("## 📂 PDF Library")

    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder not found:\n`{DATA_FOLDER}`")
        pdf_files = []
    else:
        pdf_files = [
            f for f in os.listdir(DATA_FOLDER)
            if f.lower().endswith(".pdf")
        ]
        if not pdf_files:
            st.warning("No PDFs found in `/data`.\nAdd some `.pdf` files and refresh.")
        else:
            for f in pdf_files:
                st.markdown(f"- `{f}`")

    st.markdown("---")
    if st.button("🔁 Rebuild index (all PDFs)"):
        # Force rebuild next time main area runs
        for key in ["gguf_store", "messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Index will be rebuilt from PDFs on next message.")


# --------------------------
# MAIN AREA – Chat UI
# --------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #050816;
    }
    .stChatMessage {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🦙 Local Multi-PDF Chat (LLaMA 3.1 8B GGUF)")
st.caption("Chat with **all PDFs in `/data`** using a fully local LLaMA model. No API keys, no cloud.")

# Session state: vector store + chat history
if "gguf_store" not in st.session_state:
    if not os.path.exists(DATA_FOLDER):
        st.error(f"Cannot build index – folder missing:\n`{DATA_FOLDER}`")
    elif not pdf_files:
        st.warning("Add some PDFs to `/data` first.")
    else:
        with st.spinner("📚 Indexing all PDFs into a vector store (one-time per session)..."):
            store, chunks = build_vector_store_from_folder_gguf(DATA_FOLDER)
        st.session_state.gguf_store = store
        st.session_state.messages = []
        st.success("✅ Index built! You can start chatting.")

# Initialise messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar="🧑" if msg["role"] == "user" else "🦙"):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something about your PDFs...")

if user_input and user_input.strip():
    if "gguf_store" not in st.session_state:
        st.error("Vector store not ready. Check sidebar or restart the app.")
    else:
        # 1) Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Immediately show it
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # 2) Get answer from RAG + LLaMA
        with st.chat_message("assistant", avatar="🦙"):
            with st.spinner("Thinking with your PDFs + LLaMA 3.1 8B..."):
                answer = answer_question_multi_pdf_gguf(
                    st.session_state.gguf_store,
                    user_input,
                    top_k=5,
                )
            st.markdown(answer)

        # 3) Save assistant reply to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
