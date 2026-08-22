"""Guardrails module for safe and grounded AI responses."""
import re
from typing import List, Tuple, Dict

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

def _tokenize(text: str) -> set:
    """Extract lowercase word tokens."""
    return set(re.findall(r'\w+', text.lower()))

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
        return True, 1.0

    overlap = len(ans_unigrams.intersection(ctx_unigrams)) + len(ans_bigrams.intersection(ctx_bigrams))
    overlap_ratio = overlap / total_ans_ngrams

    return overlap_ratio >= 0.08, overlap_ratio


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def map_evidence(answer: str, sources: List[Dict]) -> List[Dict]:
    """For each sentence in the answer, find the best-matching source.
    
    Returns a list of dicts: { sentence, source_idx, confidence, source_id }
    """
    if not answer or not sources:
        return []

    sentences = split_sentences(answer)
    evidence_map = []

    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            continue

        best_idx = -1
        best_overlap = 0.0

        for i, src in enumerate(sources):
            src_tokens = _tokenize(src.get("text", ""))
            if not src_tokens:
                continue
            overlap = len(sent_tokens & src_tokens) / max(len(sent_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        confidence = round(best_overlap, 3)
        evidence_map.append({
            "sentence": sent,
            "source_idx": best_idx,
            "source_id": f"[{best_idx + 1}]" if best_idx >= 0 else "[?]",
            "confidence": confidence,
        })

    return evidence_map


def compute_evidence_score(evidence_map: List[Dict]) -> float:
    """Compute an aggregate confidence score from evidence mapping."""
    if not evidence_map:
        return 0.0
    scores = [e["confidence"] for e in evidence_map]
    return round(sum(scores) / len(scores), 3)


def check_refusal(answer: str) -> bool:
    """Check if the model refused to answer or indicated lack of knowledge."""
    refusal_patterns = [
        r"i'?m not aware",
        r"i don'?t (?:have|know|recall)",
        r"i am not (?:aware|familiar)",
        r"doesn'?t include",
        r"not (?:include|contain|available)",
        r"no relevant context",
        r"outside the knowledge base",
        r"may need to consult",
        r"i cannot process",
        r"not (?:found|present) in",
        r"don'?t have information",
    ]
    lower = answer.lower()
    return any(re.search(p, lower) for p in refusal_patterns)
