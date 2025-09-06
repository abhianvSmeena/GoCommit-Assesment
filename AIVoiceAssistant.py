# AIVoiceAssistant.py
"""
AIVoiceAssistant with:
 - Hybrid retrieval (Chroma vector + TF-IDF)
 - Chroma compatibility fallbacks
 - UTF-8 subprocess calls to Ollama
 - In-memory hybrid query cache (simple TTL + max items)
 - Page-aware PDF ingestion + OCR fallback
 - Sentiment analysis (optional)
"""
import os
import json
import time
import logging
import pathlib
import subprocess
import re
from typing import List, Dict, Iterator, Tuple

from langchain.text_splitter import CharacterTextSplitter
import PyPDF2

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# OCR optionals
try:
    from pdf2image import convert_from_path
    _HAS_PDF2IMAGE = True
except Exception:
    _HAS_PDF2IMAGE = False

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

# Sentiment optional
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# TF-IDF optional
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

import config

logger = logging.getLogger("AIVoiceAssistant")
logging.basicConfig(level=logging.INFO)


def extract_text_pages_from_pdf(path: str) -> Iterator[Tuple[int, str]]:
    p = pathlib.Path(path)
    if not p.exists():
        return
    try:
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                yield (i + 1, txt)
    except Exception as e:
        logger.exception("Failed reading PDF %s with PyPDF2: %s", path, e)
        return


def ocr_pdf_pages(path: str, dpi: int = 300, poppler_path: str = None) -> Iterator[Tuple[int, str]]:
    if not _HAS_PDF2IMAGE or not _HAS_PYTESSERACT:
        raise RuntimeError("OCR dependencies not available (pdf2image and pytesseract required).")
    p = pathlib.Path(path)
    if not p.exists():
        return
    poppler_path = poppler_path or os.getenv("POPPLER_PATH")
    try:
        images = convert_from_path(path, dpi=dpi, poppler_path=poppler_path)
    except Exception as e:
        logger.exception("pdf2image.convert_from_path failed for %s: %s", path, e)
        return
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img)
        except Exception as e:
            logger.warning("pytesseract failed on page %d of %s: %s", i + 1, path, e)
            text = ""
        yield (i + 1, text)


class ChromaKB:
    def __init__(self, persist_directory: str = config.CHROMA_PERSIST_DIR, model_name: str = config.EMBED_MODEL):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Embedding wrapper (SentenceTransformer via chroma utils)
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        except Exception:
            self.embedding_function = None

        # Try modern Settings then fallback
        try:
            try:
                settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory)
                self.client = chromadb.Client(settings=settings)
            except TypeError:
                settings = Settings(persist_directory=self.persist_directory)
                self.client = chromadb.Client(settings=settings)
        except Exception as e:
            logger.warning("Chroma Settings init failed: %s — falling back to chromadb.Client()", e)
            self.client = chromadb.Client()

        # sanitize collection name
        name = os.getenv("CHROMA_COLLECTION_NAME", getattr(config, "CHROMA_COLLECTION_NAME", "kb_collection"))
        sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", str(name))
        if len(sanitized) < 3:
            sanitized = (sanitized + "col")[:3]
        if not sanitized[0].isalnum():
            sanitized = "c" + sanitized[1:]
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + "c"
        self.collection_name = sanitized

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info("Got existing Chroma collection '%s'", self.collection_name)
        except Exception:
            # create collection with embedding function if available
            if self.embedding_function:
                self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self.embedding_function)
            else:
                self.collection = self.client.create_collection(name=self.collection_name)
            logger.info("Created Chroma collection '%s'", self.collection_name)

    def add_documents(self, docs: List[Dict]):
        if not docs:
            return
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
        try:
            self.client.persist()
        except Exception:
            logger.debug("Chroma persist() not available/failed (non-fatal).")
        logger.info("Chroma: added %d docs", len(docs))

    def query_vector(self, query: str, top_k: int = 10):
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.exception("Chroma query failed: %s", e)
            return []
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])
        ids = ids[0] if ids else [None] * len(documents)
        out = []
        for i in range(len(documents)):
            out.append({
                "id": ids[i] if i < len(ids) else None,
                "text": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": float(distances[i]) if i < len(distances) else 0.0
            })
        return out


class AIVoiceAssistant:
    def __init__(self):
        # config
        self.kb_dir = pathlib.Path(os.getenv("KB_DIR", config.KB_DIR))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", config.CHUNK_SIZE))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", config.CHUNK_OVERLAP))
        self.top_k = int(os.getenv("TOP_K", config.TOP_K))
        self.hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.7"))

        # KBs
        self.kb = ChromaKB()

        # TF-IDF manifest & index
        self.documents_manifest_path = pathlib.Path(self.kb.persist_directory) / "documents.json"
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.local_docs = []

        # text splitter
        self.splitter = CharacterTextSplitter(separator="\n", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len)

        # sentiment
        self.sentiment = None
        if config.USE_SENTIMENT and _HAS_TRANSFORMERS:
            try:
                self.sentiment = pipeline("sentiment-analysis")
                logger.info("Loaded sentiment pipeline.")
            except Exception as e:
                logger.warning("Sentiment pipeline load failed: %s", e)

        # LLM / Ollama config
        self.llm_backend = os.getenv("LLM_BACKEND", config.LLM_BACKEND)
        self.llm_model = os.getenv("LLM_MODEL", config.LLM_MODEL)
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", config.LLM_MAX_TOKENS))
        self.llm_timeout = float(os.getenv("LLM_TIMEOUT", config.LLM_TIMEOUT))

        # system prompt
        self.system_prompt = (
            "You are a helpful, concise AI assistant. Use the CONTEXT provided. "
            "Be empathetic if user is distressed. At the end, add a 'Sources:' list with file paths and page numbers."
        )

        # Simple in-memory cache for hybrid_query results
        self._hybrid_cache = {}  # {qkey: (timestamp, results)}
        self._hybrid_cache_ttl = float(os.getenv("HYBRID_CACHE_TTL", "300.0"))
        self._hybrid_cache_max_items = int(os.getenv("HYBRID_CACHE_MAX_ITEMS", "256"))

        # load or ingest docs -> build TF-IDF
        if self.documents_manifest_path.exists():
            self._load_local_docs_and_build_tfidf()
        else:
            self._ingest_and_build()

    # Ingestion
    def _ingest_and_build(self):
        if not self.kb_dir.exists():
            logger.warning("KB dir %s missing", self.kb_dir)
            return

        files = [p for p in self.kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".pdf", ".md", ".txt")]
        docs_to_add = []
        local_docs = []

        for f in files:
            try:
                if f.suffix.lower() == ".pdf":
                    any_text = False
                    for page_no, page_text in extract_text_pages_from_pdf(str(f)):
                        if page_text and page_text.strip():
                            any_text = True
                            chunks = self.splitter.split_text(page_text)
                            for idx, chunk in enumerate(chunks):
                                doc_id = f"{f.name}-p{page_no}-{idx}"
                                md = {"source": str(f.name), "page": page_no}
                                docs_to_add.append({"id": doc_id, "text": chunk, "metadata": md})
                                local_docs.append({"id": doc_id, "text": chunk, "metadata": md})
                    if not any_text and _HAS_PDF2IMAGE and _HAS_PYTESSERACT:
                        poppler = os.getenv("POPPLER_PATH", None)
                        try:
                            for pno, ocr_text in ocr_pdf_pages(str(f), poppler_path=poppler):
                                if not ocr_text or not ocr_text.strip():
                                    continue
                                chunks = self.splitter.split_text(ocr_text)
                                for idx, chunk in enumerate(chunks):
                                    doc_id = f"{f.name}-p{pno}-ocr-{idx}"
                                    md = {"source": str(f.name), "page": pno}
                                    docs_to_add.append({"id": doc_id, "text": chunk, "metadata": md})
                                    local_docs.append({"id": doc_id, "text": chunk, "metadata": md})
                        except Exception as e:
                            logger.exception("OCR failed for %s: %s", f, e)
                else:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    if text and text.strip():
                        chunks = self.splitter.split_text(text)
                        for idx, chunk in enumerate(chunks):
                            doc_id = f"{f.name}-{idx}"
                            md = {"source": str(f.name)}
                            docs_to_add.append({"id": doc_id, "text": chunk, "metadata": md})
                            local_docs.append({"id": doc_id, "text": chunk, "metadata": md})
            except Exception as e:
                logger.exception("Failed ingesting %s: %s", f, e)

        if docs_to_add:
            BATCH = 256
            for i in range(0, len(docs_to_add), BATCH):
                self.kb.add_documents(docs_to_add[i:i+BATCH])
            logger.info("Ingested %d chunks into Chroma.", len(docs_to_add))
        else:
            logger.warning("No documents ingested from %s", self.kb_dir)

        # save manifest & build tfidf
        self.local_docs = local_docs
        try:
            with open(self.documents_manifest_path, "w", encoding="utf-8") as fh:
                json.dump(self.local_docs, fh, ensure_ascii=False, indent=2)
            logger.info("Saved documents manifest to %s", self.documents_manifest_path)
        except Exception as e:
            logger.exception("Failed to save manifest: %s", e)

        self._build_tfidf_index()

    def _load_local_docs_and_build_tfidf(self):
        try:
            with open(self.documents_manifest_path, "r", encoding="utf-8") as fh:
                self.local_docs = json.load(fh)
            logger.info("Loaded %d local docs from manifest.", len(self.local_docs))
        except Exception as e:
            logger.warning("Failed to load manifest: %s — will re-ingest.", e)
            self._ingest_and_build()
            return
        self._build_tfidf_index()

    def _build_tfidf_index(self):
        if not _HAS_SKLEARN:
            logger.warning("scikit-learn not installed; TF-IDF disabled.")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
        texts = [d.get("text", "") for d in self.local_docs]
        if not texts:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            logger.info("No texts to TF-IDF index.")
            return
        try:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=32768, ngram_range=(1, 2), stop_words="english")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("Built TF-IDF index shape %s", self.tfidf_matrix.shape)
        except Exception as e:
            logger.exception("TF-IDF build failed: %s", e)
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    # Hybrid retrieval
    def _compute_tfidf_scores(self, query: str):
        if not _HAS_SKLEARN or self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        try:
            qv = self.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(self.tfidf_matrix, qv).reshape(-1)
            return sims.tolist()
        except Exception as e:
            logger.exception("TF-IDF scoring failed: %s", e)
            return []

    def hybrid_query(self, query: str, top_k: int = None, alpha: float = None):
        top_k = top_k or self.top_k
        alpha = alpha if alpha is not None else self.hybrid_alpha

        # cache check
        qkey = query.strip().lower()
        now = time.time()
        if len(self._hybrid_cache) > self._hybrid_cache_max_items:
            items = sorted(self._hybrid_cache.items(), key=lambda x: x[1][0])
            for k, _ in items[: max(1, len(items)//10) ]:
                self._hybrid_cache.pop(k, None)
        cached = self._hybrid_cache.get(qkey)
        if cached:
            ts, cached_results = cached
            if now - ts <= self._hybrid_cache_ttl:
                return [dict(r) for r in cached_results][: top_k]

        # vector hits (get more for rerank)
        vec_hits = self.kb.query_vector(query, top_k=top_k * 3)
        for h in vec_hits:
            dist = h.get("distance", 0.0)
            h["vec_sim"] = 1.0 / (1.0 + dist) if dist >= 0 else 0.0

        # tfidf scores
        tfidf_scores = self._compute_tfidf_scores(query)
        id_to_tfidf = {}
        if tfidf_scores and self.local_docs:
            for idx, score in enumerate(tfidf_scores):
                did = self.local_docs[idx].get("id")
                id_to_tfidf[did] = float(score)

        candidates = {}
        for h in vec_hits:
            did = h.get("id")
            candidates[did] = {
                "id": did,
                "text": h.get("text"),
                "metadata": h.get("metadata", {}),
                "vec_sim": h.get("vec_sim", 0.0),
                "tfidf_sim": id_to_tfidf.get(did, 0.0)
            }

        if id_to_tfidf:
            sorted_tfidf = sorted(id_to_tfidf.items(), key=lambda x: x[1], reverse=True)[: top_k * 3]
            for did, score in sorted_tfidf:
                if did in candidates:
                    continue
                doc = next((d for d in self.local_docs if d.get("id") == did), None)
                if doc:
                    candidates[did] = {
                        "id": did,
                        "text": doc.get("text"),
                        "metadata": doc.get("metadata", {}),
                        "vec_sim": 0.0,
                        "tfidf_sim": float(score)
                    }

        results = []
        for did, v in candidates.items():
            vec_s = v.get("vec_sim", 0.0)
            tfidf_s = v.get("tfidf_sim", 0.0)
            combined = alpha * vec_s + (1.0 - alpha) * tfidf_s
            v["combined_score"] = float(combined)
            results.append(v)
        results = sorted(results, key=lambda x: x["combined_score"], reverse=True)[:top_k]

        # cache
        try:
            self._hybrid_cache[qkey] = (now, [dict(r) for r in results])
        except Exception:
            pass

        return results

    # context + citation construction
    def _construct_context_and_citations(self, query: str, top_k: int = None):
        top_k = top_k or self.top_k
        hits = self.hybrid_query(query, top_k=top_k)
        pieces = []
        sources = []
        seen = set()
        for i, h in enumerate(hits):
            md = h.get("metadata", {}) or {}
            src = md.get("source", "unknown")
            page = md.get("page")
            src_label = str(src) if page is None else f"{src} (page {page})"
            snippet = (h.get("text", "") or "")[:1200].strip()
            pieces.append(f"[SOURCE {i+1}: {src_label} | score={h.get('combined_score'):.4f}]\n{snippet}\n")
            if (src, page) not in seen:
                sources.append(src_label)
                seen.add((src, page))
        context = "\n---\n".join(pieces)
        return context, sources, hits

    # Ollama caller (UTF-8)
    def _call_ollama_cli(self, prompt: str) -> str:
        last_errs = []
        cmds_to_try = [
            (["ollama", "run", self.llm_model], True),
            (["ollama", "predict", self.llm_model], True),
            (["ollama", "run", self.llm_model, prompt], False),
            (["ollama", "predict", self.llm_model, prompt], False),
            (["ollama", "generate", self.llm_model, prompt], False),
        ]

        for cmd, use_stdin in cmds_to_try:
            try:
                if use_stdin:
                    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=max(30, int(self.llm_timeout)))
                else:
                    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=max(30, int(self.llm_timeout)))
            except FileNotFoundError:
                return "Ollama CLI not found. Install Ollama and ensure 'ollama' is in PATH: https://ollama.com"
            except Exception as e:
                last_errs.append(f"Run failed {' '.join(cmd)}: {e}")
                continue

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            rc = proc.returncode

            if stdout:
                return stdout
            if rc == 0 and stderr:
                return stderr
            if stderr:
                last_errs.append(f"Cmd {' '.join(cmd)} rc={rc} stderr={stderr}")
            else:
                last_errs.append(f"Cmd {' '.join(cmd)} rc={rc} no output")

        err_text = "\n".join(last_errs) if last_errs else "Ollama call failed."
        logger.warning("Ollama attempts exhausted. Details:\n%s", err_text)
        return f"Error generating response via Ollama. Details:\n{err_text}"

    def analyze_sentiment(self, text: str):
        if not self.sentiment:
            return {"label": "NEUTRAL", "score": 0.0}
        try:
            out = self.sentiment(text, truncation=True)
            if isinstance(out, list) and out:
                return {"label": out[0]["label"], "score": float(out[0]["score"])}
        except Exception as e:
            logger.debug("Sentiment error: %s", e)
        return {"label": "NEUTRAL", "score": 0.0}

    def interact_with_llm(self, user_query: str):
        start = time.time()
        sentiment = self.analyze_sentiment(user_query)
        tone_hint = ""
        if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.6:
            tone_hint = "User seems upset. Answer empathetically and concisely."
        elif sentiment["label"] == "POSITIVE" and sentiment["score"] > 0.6:
            tone_hint = "User seems positive; keep tone upbeat."

        context, sources, hits = self._construct_context_and_citations(user_query, top_k=self.top_k)

        prompt_parts = [f"SYSTEM: {self.system_prompt}"]
        if tone_hint:
            prompt_parts.append(f"NOTE: {tone_hint}")
        if context:
            prompt_parts.append(f"CONTEXT:\n{context}")
        prompt_parts.append(f"USER: {user_query}")
        prompt_parts.append("\nAnswer concisely and append a 'Sources:' list with file paths and page numbers at the end.")
        final_prompt = "\n\n".join(prompt_parts)

        try:
            if self.llm_backend == "ollama":
                answer = self._call_ollama_cli(final_prompt)
            else:
                answer = "LLM backend not configured."
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            answer = "Error generating answer."

        latency = time.time() - start

        return {
            "answer": answer,
            "sources": sources,
            "sentiment": sentiment,
            "latency": latency,
            "hits": hits
        }

