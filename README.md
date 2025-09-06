# Voice-Enabled RAG Assistant with Sentiment-Aware Response  

## Overview  
This project is a **voice-enabled Retrieval Augmented Generation (RAG) assistant** that combines **speech recognition, hybrid document retrieval, sentiment-aware response generation, and text-to-speech output** into one modular system.  

Built with **local models** (Whisper, SentenceTransformers, Gemma via Ollama), it allows **offline execution** while remaining extensible to other LLM backends or vector databases.  

Pipeline at a glance:  

🎤 **Voice Input** → 📝 **Transcription (Whisper)** → 📚 **Hybrid Retrieval (Chroma + TF-IDF)** → 🧠 **Context Injection + Sentiment Analysis** → 🤖 **LLM Response (Gemma via Ollama)** → 🔊 **Voice Output (TTS)** 

Demonstration Video:
https://drive.google.com/file/d/189NOed7-5-UJ5pgJvXWUwqGVCz-AV1tL/view?usp=sharing

---

## Features  
- **Multi-mode Input**  
  - Text query (keyboard input)  
  - Audio file input (`.wav`)  
  - Live microphone recording (press-to-start / press-to-stop)  

- **Document Ingestion & Knowledge Base**  
  - Supports **Markdown (`.md`)**, **Text (`.txt`)**, and **PDFs**  
  - Uses **PyPDF2** with **OCR fallback (pdf2image + pytesseract)** for scanned PDFs  
  - Splits documents into context-aware chunks  

- **Hybrid Retrieval**  
  - Vector similarity search with **SentenceTransformers (`all-MiniLM-L6-v2`)** + **ChromaDB**  
  - Keyword-based retrieval using **TF-IDF + cosine similarity**  
  - Combined scoring ensures robust semantic + keyword retrieval  

- **Sentiment-Aware Generation**  
  - HuggingFace **sentiment pipeline** injects tone hints into system prompt  
  - Negative sentiment → empathetic responses  
  - Positive sentiment → upbeat, encouraging responses  

- **LLM Backend**  
  - Runs **Gemma:2B** locally via **Ollama CLI**  
  - Configurable to swap with other Ollama-supported LLMs  

- **Voice Output (TTS)**  
  - Offline speech synthesis with **pyttsx3**  
  - Fallback to **gTTS + pygame** for natural voice  
  - Cleans up Markdown/citations before playback  

- **Performance Enhancements**  
  - **In-memory hybrid retrieval cache** (reduces repeated query latency)  
  - Configurable **chunk size, retrieval top-k, token limits** via `config.py` or env vars  

---

PROJECT STRUCTURE
```
.
├── app.py # Main entrypoint: mic/file/text input modes
├── AIVoiceAssistant.py # RAG pipeline, retrieval, LLM, sentiment analysis 
├── voice_service.py # TTS engine (pyttsx3 / gTTS fallback)
├── config.py # Centralized project configuration
├── build_kb.py # Document ingestion + KB building
├── rag/
│ └── docs/ # Knowledge base documents (PDF/MD/TXT)
├── data/
│ └── chroma_db/ # Persistent Chroma database
└── requirements.txt     # Core dependencies
```
QUICK START

1. Clone the repo
```
https://github.com/abhianvSmeena/GoCommit-Assesment.git
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Create a virtual env file:
```
python -m venv venv
```

4. Ingest Data File:
```
python build_kb.py
```

5. Start the LLM:
```
python app.py
```
