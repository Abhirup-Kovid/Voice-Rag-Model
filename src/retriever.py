"""Retriever module backed by Pinecone serverless hybrid search."""
import time
import structlog
from typing import List, Optional
from src.config import settings
from src.schemas import Source
from src.embedder import embed_texts
from src import pinecone_store

logger = structlog.get_logger(__name__)

def warmup():
    """Prime ONNX session + Pinecone index host to eliminate cold-start latency."""
    try:
        from src.embedder import warmup as embedder_warmup
        embedder_warmup()
        pinecone_store.load_vocab()
        pinecone_store.warmup()
        logger.info("Retriever warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

def retrieve(query: str, language: Optional[str] = None, k: int = 8) -> List[Source]:
    """Retrieve top-k chunks via hybrid search and rerank with a language bonus."""
    t_start = time.time()

    # 1. Embed query locally (same model as corpus -> aligned vector space)
    query_emb = embed_texts([query], is_query=True)[0]
    t_embed = time.time()

    # 2. Hybrid query (dense + client-side sparse), fetch extra candidates
    sparse = pinecone_store.build_sparse(query)
    resp = pinecone_store.hybrid_query(query_emb, sparse, top_k=max(k * 4, 24))
    t_query = time.time()

    # 3. Rerank: Pinecone score + language bonus
    scored = []
    for m in resp.matches:
        meta = m.get("metadata", {})
        lang_bonus = 0.10 if language and meta.get("language") == language else 0.0
        scored.append((m.score + lang_bonus, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    sources = []
    for score, m in scored[:settings.TOP_K_FINAL]:
        meta = m.get("metadata", {})
        sources.append(Source(
            chunk_id=m.get("id", ""),
            text=meta.get("text", ""),
            language=meta.get("language", ""),
            strategy=meta.get("strategy", "hybrid"),
            score=score,
            doc_id=meta.get("doc_id", ""),
            char_offsets=[0, 0]
        ))

    t_end = time.time()
    logger.info(
        f"retrieve phases: embed {(t_embed - t_start) * 1000:.0f}ms | "
        f"pinecone {(t_query - t_embed) * 1000:.0f}ms | "
        f"rerank {(t_end - t_query) * 1000:.0f}ms | total {(t_end - t_start) * 1000:.0f}ms"
    )
    return sources