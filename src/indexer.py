"""Indexer module for FAISS and BM25."""
import os
import json
import pickle
import faiss
import numpy as np
import structlog
import gc
from typing import List, Dict, Tuple, Any
from rank_bm25 import BM25Okapi
from src.config import settings
from src.schemas import Chunk

logger = structlog.get_logger(__name__)

_faiss_index = None
_bm25_index = None
_chunks_metadata = None
_topic_centroid = None
_passage_map = None

def build_index(chunks: List[Chunk], embeddings: np.ndarray):
    """Build and save FAISS FlatIP, BM25, and Topic Centroid."""
    logger.info("Building indices...")
    os.makedirs(settings.INDEX_DIR, exist_ok=True)
    
    # 1. FAISS FlatIP (exact inner product search)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
        
    faiss.write_index(index, os.path.join(settings.INDEX_DIR, "faiss_ivf.index"))
    
    # 2. Metadata
    metadata = [c.model_dump() for c in chunks]
    with open(os.path.join(settings.INDEX_DIR, "chunks_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
        
    # 3. BM25
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(settings.INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
        
    # 4. Topic Centroid
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    np.save(os.path.join(settings.INDEX_DIR, "topic_centroid.npy"), centroid)
    
    logger.info("Indices built successfully.")

def load_index() -> Tuple[Any, Any, List[Dict], np.ndarray]:
    """Load indices into memory."""
    global _faiss_index, _bm25_index, _chunks_metadata, _topic_centroid
    
    if _faiss_index is not None:
        return _faiss_index, _bm25_index, _chunks_metadata, _topic_centroid
        
    faiss_path = os.path.join(settings.INDEX_DIR, "faiss_ivf.index")
    meta_path = os.path.join(settings.INDEX_DIR, "chunks_metadata.json")
    bm25_path = os.path.join(settings.INDEX_DIR, "bm25.pkl")
    centroid_path = os.path.join(settings.INDEX_DIR, "topic_centroid.npy")
    
    if not os.path.exists(faiss_path):
        logger.warning("Index files not found. Run index builder first.")
        return None, None, None, None
        
    logger.info("Loading indices...")
    _faiss_index = faiss.read_index(faiss_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        _chunks_metadata = json.load(f)
        
    with open(bm25_path, "rb") as f:
        _bm25_index = pickle.load(f)
        
    _topic_centroid = np.load(centroid_path)
    
    logger.info("Indices loaded.")
    return _faiss_index, _bm25_index, _chunks_metadata, _topic_centroid

def get_loaded_indices():
    return load_index()

def get_passage_map() -> dict:
    """Lazy-load doc_id -> full passage text map (used to expand short chunks)."""
    global _passage_map
    if _passage_map is not None:
        return _passage_map
    _passage_map = {}
    path = os.path.join(settings.DATA_DIR, "processed_passages.jsonl")
    if not os.path.exists(path):
        logger.warning("Passage file not found; short chunks will not be expanded.")
        return _passage_map
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                _passage_map[row["doc_id"]] = row["text"]
        logger.info(f"Loaded passage map: {len(_passage_map)} entries")
    except Exception as e:
        logger.warning(f"Failed to load passage map: {e}")
    return _passage_map
