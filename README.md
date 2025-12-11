🚀 Local Multi-PDF LLaMA Assistant
A Fully Offline RAG System Using LLaMA, FAISS, and Sentence Transformers
🧠 Overview

Local Multi-PDF LLaMA Assistant is a 100% offline, privacy-focused RAG (Retrieval-Augmented Generation) system that:

Reads multiple PDFs

Splits them into chunks

Generates semantic embeddings

Performs similarity search with FAISS

Feeds retrieved context into a local LLaMA GGUF model

Answers questions like ChatGPT — but completely offline

This project demonstrates practical, industry-level AI engineering skills, including NLP pipelines, embedding models, vector databases, and local LLM inference.

🌟 Key Features
📚 Multi-PDF Support

Automatically loads every PDF in the data/ folder.

✂️ Smart Text Chunking

Chunking with overlap for maximum context retention.

🔍 Semantic Embeddings

Using sentence-transformers/all-MiniLM-L6-v2.

⚡ FAISS Vector Search

Fast similarity queries across thousands of chunks.

🤖 Local LLaMA GGUF Model

Runs entirely offline using llama-cpp-python.

No API keys.
No internet.
No privacy risk.

💬 ChatGPT-Style Streamlit UI

Chat bubbles

Message history

Dark mode

Sidebar showing indexed PDFs

Smooth UX

🚀 GPU Acceleration (Optional)

Automatically uses CUDA if installed.

🧩 Modular Architecture

Every component cleanly separated inside modules/.

🏗️ Architecture Diagram
PDFs → Text Extraction → Chunking → Embeddings → FAISS Search → Top-K Context
                  ↓                                                ↑
                  └────────────── LLaMA GGUF Model ←───────────────┘

📁 Project Structure
local-multi-pdf-llama-assistant/
 ┣ modules/
 ┃ ┣ pdf_loader.py
 ┃ ┣ text_splitter.py
 ┃ ┣ embedder.py
 ┃ ┣ vector_store.py
 ┃ ┣ local_llm.py
 ┃ ┣ local_llm_gguf.py
 ┃ ┣ multi_pdf_loader.py
 ┃ ┗ multi_rag.py
 ┣ data/
 ┣ models/
 ┣ app.py
 ┣ app_chat_gguf.py
 ┣ app_gguf.py
 ┣ README.md
 ┣ requirements.txt
 ┣ .gitignore

🔧 Installation
1️⃣ Clone repo
git clone https://github.com/swarajshinde12/local-multi-pdf-llama-assistant
cd local-multi-pdf-llama-assistant

2️⃣ Create virtual environment
python -m venv venv
.\venv\Scripts\activate

3️⃣ Install requirements
pip install -r requirements.txt

4️⃣ Add a GGUF model

Download any LLaMA or Mistral GGUF file (example):

Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

Place it in:

models/llm.gguf

▶️ Run the App
Chat Interface (recommended)
streamlit run app_chat_gguf.py

Basic RAG app
streamlit run app_gguf.py

💡 Example Query

User:

What does this document say about neural networks?

Assistant (local LLaMA):
Summarizes using retrieved chunks + LLM reasoning.

🎯 Why This Project Impresses Recruiters

This project shows you can:

✔ Implement real RAG pipelines
✔ Work with embeddings + FAISS
✔ Run local LLMs with quantization
✔ Build modular AI systems
✔ Build clean UI apps
✔ Handle multi-PDF knowledge bases
✔ Optimize for GPU where possible

This is exactly what companies hiring ML/AI engineers look for.

🔮 Future Enhancements

Add reranking (BGE-Reranker / ColBERT)

Add conversation memory

Show citations in responses

Improve UI animations

Add support for DOCX / TXT
