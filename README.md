# Local Multi-PDF LLaMA Assistant  
A fully offline Retrieval-Augmented Generation (RAG) system built using LLaMA (GGUF), FAISS, and Sentence Transformers.  
This project demonstrates a complete end-to-end AI pipeline capable of indexing multiple PDFs and answering queries using a local large language model — without any internet or API keys.

---

## Overview

The Local Multi-PDF LLaMA Assistant is designed as a privacy-focused, offline question-answering system.  
It performs text extraction from PDFs, chunking, embedding generation, vector similarity search, and final answer generation through a locally hosted LLaMA model.

This system reflects industry-grade concepts used in enterprise RAG applications, including vector databases, embedding architectures, context-aware LLM prompting, and scalable document ingestion.

---

## Key Features

### 1. Multi-PDF Document Ingestion  
Automatically loads and processes every PDF placed in the `data/` directory.

### 2. Text Chunking Pipeline  
Implements sliding-window chunking to preserve semantic continuity and increase retrieval accuracy.

### 3. Embedding Generation  
Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text chunks into dense vectors.

### 4. Vector Search with FAISS  
Efficient nearest-neighbor retrieval across thousands of chunks.

### 5. Local LLaMA GGUF Inference  
Runs entirely offline using `llama-cpp-python`.  
Supports CPU and optional GPU acceleration (CUDA, if available).

### 6. Streamlit User Interface  
- Chat-style interface  
- Conversation history  
- Clean layout  
- Sidebar listing indexed PDFs  
- Dark mode support  

### 7. Modular Architecture  
Every major component is decoupled and placed inside the `modules/` directory for clarity and reusability.

---

## System Architecture

