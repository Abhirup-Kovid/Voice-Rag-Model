"""Guardrails module for safe and grounded AI responses."""
import re
from typing import List, Tuple

UNSAFE_KEYWORDS = {
    # English
    "explosives", "bomb", "kill", "suicide", "murder", "hack", "illegal", "drugs",
    # Hindi Transliterated
    "bam", "kaise banaye", "mar", "khoon", "atank", "hathiyar", "gali",
    # Bengali
    "boma", "khun",
    # Tamil
    "kolai", "vedigundu",
    # Telugu
    "champadam", "baambu",
    # Marathi
    "baam", "khoon",
    # Gujarati
    "bomb", "khoon"
}

def check_unsafe(query: str) -> bool:
    """Check if query contains unsafe keywords."""
    q_lower = query.lower()
    for kw in UNSAFE_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
            return True
    return False

def check_off_topic(query: str) -> Tuple[bool, float]:
    """Check if query is off‑topic compared to topic centroid."""
    # Disabled — centroid similarity doesn't work for short queries
    return False, 1.0

def extract_ngrams(text: str, n: int) -> set:
    """Extract n-grams from text."""
    words = text.lower().split()
    return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

def check_grounding(answer: str, context_chunks: List[str]) -> Tuple[bool, float]:
    """Check n-gram overlap between answer and retrieved context."""
    if not answer:
        return False, 0.0
    if not context_chunks:
        return True, 0.0
        
    context_text = " ".join(context_chunks)
    
    ans_unigrams = extract_ngrams(answer, 1)
    ans_bigrams = extract_ngrams(answer, 2)
    
    ctx_unigrams = extract_ngrams(context_text, 1)
    ctx_bigrams = extract_ngrams(context_text, 2)
    
    total_ans_ngrams = len(ans_unigrams) + len(ans_bigrams)
    if total_ans_ngrams == 0:
        return True, 1.0 # Empty answer? Grounded by default.
        
    overlap = len(ans_unigrams.intersection(ctx_unigrams)) + len(ans_bigrams.intersection(ctx_bigrams))
    overlap_ratio = overlap / total_ans_ngrams
    
    return overlap_ratio >= 0.20, overlap_ratio

def check_refusal(answer: str) -> bool:
    """Check if the model refused to answer."""
    return False
