"""Multi-strategy chunker."""
import re
import uuid
import tiktoken
import numpy as np
import structlog
from langdetect import detect
from typing import List
from src.schemas import Chunk
from src.embedder import embed_texts

logger = structlog.get_logger(__name__)

# Tokenizer for fixed chunking
encoder = tiktoken.get_encoding("cl100k_base")

def chunk_fixed(text: str, doc_id: str, language: str, chunk_size: int = 350, overlap: int = 50) -> List[Chunk]:
    """Strategy A: Fixed-size character chunking with overlap.
    Uses characters (not tokens) to avoid mid-byte splits in Indic scripts
    and to keep exact character offsets."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk_text = text[start_idx:end_idx]

        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text,
            strategy="fixed",
            language=language,
            doc_id=doc_id,
            token_count=len(encoder.encode(chunk_text)),
            char_start=start_idx,
            char_end=end_idx,
            metadata={"strategy": "fixed", "token_count": len(encoder.encode(chunk_text)), "char_start": start_idx, "char_end": end_idx}
        ))

        start_idx += (chunk_size - overlap)
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences handling multilingual boundaries."""
    # Regex for ., !, ?, Hindi purna viram (।), Urdu/Arabic full stop (۔)
    pattern = r'(?<=[.!?।۔])\s+'
    return re.split(pattern, text.strip())

def chunk_sentence(text: str, doc_id: str, language: str) -> List[Chunk]:
    """Strategy B: Sentence-boundary grouping until 200-400 chars."""
    sentences = split_into_sentences(text)
    chunks = []
    
    current_chunk = []
    current_length = 0
    char_start = 0
    sentence_indices = []
    
    for i, sentence in enumerate(sentences):
        if not sentence:
            continue
            
        current_chunk.append(sentence)
        current_length += len(sentence)
        sentence_indices.append(i)
        
        if current_length >= 200:
            chunk_text = " ".join(current_chunk)
            char_end = char_start + len(chunk_text)
            
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                strategy="sentence",
                language=language,
                doc_id=doc_id,
                token_count=len(encoder.encode(chunk_text)),
                char_start=char_start,
                char_end=char_end,
                metadata={"strategy": "sentence", "sentence_count": len(current_chunk), "char_start": char_start, "char_end": char_end, "sentence_indices": sentence_indices}
            ))
            
            char_start = char_end + 1
            current_chunk = []
            current_length = 0
            sentence_indices = []
            
    # Add remaining
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        char_end = char_start + len(chunk_text)
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text,
            strategy="sentence",
            language=language,
            doc_id=doc_id,
            token_count=len(encoder.encode(chunk_text)),
            char_start=char_start,
            char_end=char_end,
            metadata={"strategy": "sentence", "sentence_count": len(current_chunk), "char_start": char_start, "char_end": char_end, "sentence_indices": sentence_indices}
        ))
        
    return chunks

def chunk_semantic(text: str, doc_id: str, language: str, threshold: float = 0.55) -> List[Chunk]:
    """Strategy C: Semantic chunking."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(chunk_id=str(uuid.uuid4()), text=sentences[0], strategy="semantic", language=language, doc_id=doc_id, token_count=len(encoder.encode(sentences[0])), char_start=0, char_end=len(sentences[0]), metadata={"strategy": "semantic", "semantic_boundary_score": 1.0, "char_start": 0, "char_end": len(sentences[0]), "embedding_id": ""})]
        
    embeddings = embed_texts(sentences, is_query=False)
    
    chunks = []
    current_chunk = [sentences[0]]
    char_start = 0
    
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i])
        
        if sim < threshold:
            # Boundary
            chunk_text = " ".join(current_chunk)
            char_end = char_start + len(chunk_text)
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                strategy="semantic",
                language=language,
                doc_id=doc_id,
                token_count=len(encoder.encode(chunk_text)),
                char_start=char_start,
                char_end=char_end,
                metadata={"strategy": "semantic", "semantic_boundary_score": float(sim), "char_start": char_start, "char_end": char_end, "embedding_id": ""}
            ))
            char_start = char_end + 1
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
            
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        char_end = char_start + len(chunk_text)
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text,
            strategy="semantic",
            language=language,
            doc_id=doc_id,
            token_count=len(encoder.encode(chunk_text)),
            char_start=char_start,
            char_end=char_end,
            metadata={"strategy": "semantic", "semantic_boundary_score": 1.0, "char_start": char_start, "char_end": char_end, "embedding_id": ""}
        ))
        
    return chunks

def chunk_lang_aware(text: str, doc_id: str) -> List[Chunk]:
    """Strategy D: Language-aware chunking."""
    try:
        lang = detect(text)
    except:
        lang = "en"
        
    indic_langs = ["hi", "bn", "ta", "te", "mr", "gu"]
    target_len = 200 if lang in indic_langs else 400
    
    # Just split by character length for simplicity in this strategy
    chunks = []
    for i in range(0, len(text), target_len):
        chunk_text = text[i:i+target_len]
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text,
            strategy="lang_aware",
            language=lang,
            doc_id=doc_id,
            token_count=len(encoder.encode(chunk_text)),
            char_start=i,
            char_end=i+len(chunk_text),
            metadata={"doc_id": doc_id, "language": lang, "chunk_strategy": "lang_aware"}
        ))
        
    return chunks

def process_all_strategies(text: str, doc_id: str, language: str) -> List[Chunk]:
    """Apply fixed chunking only (char-based, handles all scripts)."""
    return chunk_fixed(text, doc_id, language, chunk_size=350, overlap=50)
