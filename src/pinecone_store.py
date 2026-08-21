"""Pinecone serverless store: hybrid query + client-side sparse vocabulary."""
import json
import os
import re
import structlog
from typing import Dict, List, Optional

from src.config import settings

logger = structlog.get_logger(__name__)

_index = None
_vocab: Dict[str, int] = {}
_SPARSE_VOCAB_PATH = os.path.join(settings.DATA_DIR, "sparse_vocab.json")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

PINECONE_INDEX_HOST = "awaaz-rag-8xe0b98.svc.aped-4627-b74a.pinecone.io"


def get_index():
    global _index
    if _index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _index = pc.Index(host=PINECONE_INDEX_HOST)
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_HOST}")
    return _index


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
    """Client-side sparse vector: term counts mapped through the corpus vocab."""
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


def hybrid_query(vector: List[float], sparse: Optional[Dict[str, List]], top_k: int = 24):
    kwargs = dict(
        namespace=settings.PINECONE_NAMESPACE,
        top_k=top_k,
        vector=vector,
        include_metadata=True,
    )
    if sparse:
        kwargs["sparse_vector"] = sparse
    return get_index().query(**kwargs)


def warmup():
    """Prime the Pinecone index to absorb serverless cold start."""
    from src.embedder import embed_texts
    try:
        emb = embed_texts(["query: warmup"], is_query=True)
        if emb is not None and len(emb):
            hybrid_query(list(emb[0]), None, top_k=1)
        logger.info("Pinecone warmup complete.")
    except Exception as e:
        logger.warning(f"Pinecone warmup failed: {e}")
