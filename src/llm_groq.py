"""LLM module using Groq."""
import time
import json
import structlog
from groq import Groq
from typing import List, Dict, AsyncIterator
from src.config import settings

logger = structlog.get_logger(__name__)

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client

LANGUAGE_MAP = {
    "hi": "Hindi", "hi-IN": "Hindi", "bn": "Bengali", "bn-IN": "Bengali",
    "ta": "Tamil", "ta-IN": "Tamil", "te": "Telugu", "te-IN": "Telugu",
    "mr": "Marathi", "mr-IN": "Marathi", "gu": "Gujarati", "gu-IN": "Gujarati",
    "kn": "Kannada", "kn-IN": "Kannada", "ml": "Malayalam", "ml-IN": "Malayalam",
    "pa": "Punjabi", "pa-IN": "Punjabi", "ur": "Urdu", "ur-IN": "Urdu",
    "ne": "Nepali", "ne-IN": "Nepali", "or": "Odia", "od-IN": "Odia",
    "as": "Assamese", "as-IN": "Assamese", "en": "English", "en-IN": "English",
}

SYSTEM_PROMPT = """You are a helpful multilingual assistant for the MSMARCO dataset. Answer using the provided context when possible. If context is provided, ground your answer in it and cite sources with [1], [2] etc. If the context is not helpful, still be conversational and helpful. Be concise (1-3 sentences)."""

MAX_CHUNK_CHARS = 2500
MAX_CONTEXT_CHARS = 12000


def _build_messages(query: str, context_chunks: List[Dict], language: str) -> List[Dict]:
    lang_name = LANGUAGE_MAP.get(language, language or "English")

    if context_chunks:
        # Truncate each chunk and cap the total so the prompt never exceeds the model context window
        parts = []
        total = 0
        for i, chunk in enumerate(context_chunks):
            text = chunk["text"][:MAX_CHUNK_CHARS]
            if total + len(text) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - total
                if remaining > 200:
                    parts.append(f"[{i+1}] {text[:remaining]}")
                break
            parts.append(f"[{i+1}] {text}")
            total += len(text)
        context_text = "\n\n".join(parts)
        user_prompt = f"Context:\n{context_text}\n\nQuestion ({lang_name}): {query}\n\nIMPORTANT: You MUST reply in {lang_name}. Do NOT reply in any other language."
    else:
        user_prompt = f"Question ({lang_name}): {query}\n\nNo relevant context found. Be conversational and helpful.\n\nIMPORTANT: You MUST reply in {lang_name}. Do NOT reply in any other language."
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

def generate_stream(query: str, context_chunks: List[Dict], language: str) -> AsyncIterator[dict]:
    """Generator yielding tokens."""
    client = _get_client()
    messages = _build_messages(query, context_chunks, language)
    
    logger.info("Starting Groq streaming")
    start_time = time.time()
    first_token = True
    
    stream = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        stream=True,
        timeout=5.0
    )
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            if first_token:
                first_token_time = (time.time() - start_time) * 1000
                yield {"event": "first_token_ms", "value": int(first_token_time)}
                first_token = False
            yield {"event": "token", "text": token}
            
def generate_full(query: str, context_chunks: List[Dict], language: str) -> dict:
    """Non-streaming generate."""
    client = _get_client()
    messages = _build_messages(query, context_chunks, language)
    
    for attempt in range(2):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                stream=False,
                timeout=15.0
            )
            ttft = (time.time() - start_time) * 1000
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "is_grounded": True,
                "refused": "I don't have information about that" in answer,
                "citations": [],
                "first_token_ms": int(ttft),
                "total_ms": int(ttft)
            }
        except Exception as e:
            logger.warning(f"Groq API attempt {attempt+1} failed: {e}")
            if attempt == 1:
                raise
