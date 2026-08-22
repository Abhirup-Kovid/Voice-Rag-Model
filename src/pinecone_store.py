"""Pinecone serverless store: hybrid query via raw HTTP (bypasses SDK connection issues)."""
import json
import os
import re
import structlog
import requests
from typing import Dict, List, Optional

from src.config import settings

logger = structlog.get_logger(__name__)

_vocab: Dict[str, int] = {}
_SPARSE_VOCAB_PATH = os.path.join(settings.DATA_DIR, "sparse_vocab.json")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

PINECONE_INDEX_HOST = "awaaz-rag-8xe0b98.svc.aped-4627-b74a.pinecone.io"
PINECONE_QUERY_URL = f"https://{PINECONE_INDEX_HOST}/query"


def hybrid_query(vector: List[float], sparse: Optional[Dict[str, List]], top_k: int = 24):
    """Query Pinecone using raw HTTP POST to avoid SDK connection issues."""
    payload = {
        "namespace": settings.PINECONE_NAMESPACE,
        "topK": top_k,
        "vector": vector,
        "includeMetadata": True,
    }
    if sparse:
        payload["sparseVector"] = sparse

    headers = {
        "Api-Key": settings.PINECONE_API_KEY,
        "Content-Type": "application/json",
    }

    resp = requests.post(PINECONE_QUERY_URL, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    class Match:
        def __init__(self, m):
            self.id = m.get("id", "")
            self.score = m.get("score", 0.0)
            meta = m.get("metadata", {})
            self.text = meta.get("text", "")
            self.language = meta.get("language", "")
            self.strategy = meta.get("strategy", "")
            self.doc_id = meta.get("doc_id", "")
            self.chunk_id = meta.get("chunk_id", self.id)

    class QueryResult:
        def __init__(self, matches):
            self.matches = matches

    matches = [Match(m) for m in data.get("matches", [])]
    return QueryResult(matches)


def load_vocab() -> Dict[str, int]:
    global _vocab
    if not _vocab and os.path.exists(_SPARSE_VOCAB_PATH):
        with open(_SPARSE_VOCAB_PATH, "r", encoding="utf-8") as f:
            _vocab = json.load(f)
        logger.info(f"Loaded sparse vocab: {len(_vocab)} terms")
    return _vocab


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def build_sparse(text: str) -> Optional[Dict[str, List]]:
    vocab = load_vocab()
    if not vocab:
        return None
    counts: Dict[int, int] = {}
    for tok in tokenize(text):
        idx = vocab.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    if not counts:
        return None
    indices = sorted(counts.keys())
    return {"indices": indices, "values": [float(counts[i]) for i in indices]}


def warmup():
    from src.embedder import embed_texts
    try:
        emb = embed_texts(["query: warmup"], is_query=True)
        if emb is not None and len(emb):
            hybrid_query(list(emb[0]), None, top_k=1)
        logger.info("Pinecone warmup complete.")
    except Exception as e:
        logger.warning(f"Pinecone warmup failed: {e}")
