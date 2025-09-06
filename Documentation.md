# 📖 Documentation  

## 🔹 Assumptions  
- The system is designed to run **fully offline** using local models (Whisper, Gemma via Ollama, SentenceTransformers).  
- Chroma was chosen as the vector DB because it is lightweight, persistent, and easy to integrate, though FAISS or PGVector could be used as alternatives.  
- Whisper model size impacts latency vs. accuracy (default: `small.en`). Users can switch to `tiny.en` for speed or `medium` for accuracy.  
- The system assumes input documents are in **English**.  

---

## 🔹 Choice of Chunking Strategy / Vector DB  
- **Chunking Strategy**:  
  - Uses `CharacterTextSplitter` with **chunk size = 800 tokens** and **overlap = 200 tokens**.  
  - Balances **recall** (avoiding context loss) and **latency** (keeping prompt sizes reasonable).  
  - Larger chunks improve semantic flow, but smaller chunks increase retrieval precision.  

- **Vector DB**:  
  - **ChromaDB** is used for vector search due to its simplicity, persistence, and Python-native client.  
  - Embeddings generated with **`all-MiniLM-L6-v2` (SentenceTransformers)**.  
  - A **hybrid retrieval** approach is used:  
    - **Vector similarity** → captures semantic meaning.  
    - **TF-IDF keyword matching** → ensures exact keyword-based recall.  
    - Final ranking is a weighted combination (`alpha = 0.7` by default).  

---

## 🔹 Sentiment-Weighting Approach in Responses  
- A HuggingFace **sentiment analysis pipeline** runs on every user query.  
- Based on detected sentiment:  
  - **Negative sentiment** (high confidence): Add prompt instruction → *“Answer empathetically and concisely.”*  
  - **Positive sentiment**: Add → *“Keep tone upbeat.”*  
  - **Neutral sentiment**: No change.  
- This ensures the LLM adapts **tone/style** dynamically without retraining.  

---

## 🔹 Scalability Considerations  
- **Caching**: In-memory hybrid query cache reduces latency for repeated/related queries.  
- **Model Flexibility**:  
  - Whisper can be swapped for faster or more accurate variants.  
  - Embedding models can be replaced with larger SentenceTransformers or OpenAI embeddings (if APIs are allowed).  
- **Vector DB**: Chroma works well locally; for larger deployments, FAISS, Weaviate, or PGVector are better suited.  
- **Parallel Ingestion**: Current ingestion is sequential; can be optimized with multiprocessing for large corpora.  
- **Deployment**: Modular design allows future integration into a **web app (FastAPI/Streamlit)** instead of CLI.  
- **Hardware**: CPU-friendly by default; but GPU acceleration for Whisper + embeddings + Ollama significantly reduces latency.  
