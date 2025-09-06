# config.py
import os

KB_DIR = os.getenv("KB_DIR", "./rag/docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma:2b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))

USE_SENTIMENT = os.getenv("USE_SENTIMENT", "true").lower() in ("true","1","yes")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")  # set smaller for speed if desired
TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")

TOP_K = int(os.getenv("TOP_K", "5"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "kb_collection")

HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
HYBRID_CACHE_TTL = float(os.getenv("HYBRID_CACHE_TTL", "300.0"))
HYBRID_CACHE_MAX_ITEMS = int(os.getenv("HYBRID_CACHE_MAX_ITEMS", "256"))
