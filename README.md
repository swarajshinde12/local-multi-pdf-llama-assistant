# Local Multi-PDF LLaMA Assistant

Local RAG (Retrieval-Augmented Generation) project that:
- Reads multiple PDFs from the `data/` folder
- Creates embeddings and a FAISS vector store
- Uses a local LLaMA GGUF model via `llama-cpp-python`
- Provides a ChatGPT-style Streamlit chat UI

Code lives in `modules/` and `app_chat_gguf.py`.
Models and PDFs are **not** included in this repo.
