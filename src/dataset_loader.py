"""Dataset loader for MSMARCO-XI — loads from local cached parquet files."""
import sys
import os
import json
import glob as globmod
import argparse
from typing import List, Dict

import structlog
import pyarrow.parquet as pq

logger = structlog.get_logger(__name__)

PRIORITY_LANGS = ['hin', 'ben', 'mar', 'tam', 'tel', 'guj', 'kan', 'mal', 'pan', 'urd']


def flatten_passages(row: Dict) -> List[Dict]:
    results = []
    passages = row.get('passages', {})
    if not isinstance(passages, dict):
        return results
    eng = passages.get('English_passages', [])
    if isinstance(eng, list):
        for text in eng:
            if isinstance(text, str) and text.strip():
                results.append({'text': text.strip(), 'language': 'english'})
    trans = passages.get('Translated_passages', [])
    if isinstance(trans, list):
        for text in trans:
            if isinstance(text, str) and text.strip():
                results.append({'text': text.strip(), 'language': 'translated'})
    elif isinstance(trans, dict):
        for lang, texts in trans.items():
            if isinstance(texts, list):
                for text in texts:
                    if isinstance(text, str) and text.strip():
                        results.append({'text': text.strip(), 'language': str(lang).lower()})
    return results


def find_cached_parquets() -> dict:
    """Find all cached parquet files by language code."""
    cache_dir = "data/raw/datasets--ai4bharat--MSMARCO-XI"
    if not os.path.exists(cache_dir):
        return {}
    
    found = {}
    for parquet_path in globmod.glob(os.path.join(cache_dir, "**", "*.parquet"), recursive=True):
        basename = os.path.basename(parquet_path)
        for lang in PRIORITY_LANGS:
            if lang in basename and 'train' in basename:
                found[lang] = parquet_path
    return found


def load_parquet_file(path: str, lang: str, max_rows: int = 200000) -> List[Dict]:
    """Parse a local parquet file."""
    logger.info(f"  Loading {lang}: {path}")
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows if pf.metadata else 0
    logger.info(f"    {total_rows} rows")

    all_passages = []
    seen = set()
    rows_processed = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=1000)):
        try:
            rows = batch.to_pylist()
        except Exception:
            cols = {}
            for col_name in batch.column_names:
                try:
                    cols[col_name] = batch.column(col_name).to_pylist()
                except Exception:
                    cols[col_name] = [None] * len(batch)
            rows = []
            for i in range(len(batch)):
                row = {col_name: cols[col_name][i] for col_name in batch.column_names}
                rows.append(row)

        for idx, row in enumerate(rows):
            rows_processed += 1
            for i, p in enumerate(flatten_passages(row)):
                text = p['text']
                if text[:200] in seen:
                    continue
                seen.add(text[:200])
                query_id = row.get('query_id') or f'q{rows_processed}'
                p['doc_id'] = f"{query_id}_{i}"
                p['passage_id'] = f"passage_{len(all_passages)}"
                all_passages.append(p)
                if len(all_passages) >= max_rows:
                    break
            if len(all_passages) >= max_rows:
                break
        if len(all_passages) >= max_rows:
            break
        if batch_idx % 5 == 0:
            logger.info(f"    batch {batch_idx}: {len(all_passages)} passages")

    logger.info(f"    {rows_processed} rows -> {len(all_passages)} unique passages")
    return all_passages


def load_corpus(max_passages_per_lang: int = 150000) -> List[Dict]:
    """Load from cached parquet files (hintrain priority — has English + Hindi)."""
    cached = find_cached_parquets()
    if not cached:
        logger.error("No cached parquet files found in data/raw/")
        return []

    # Prefer hintrain if available, else ben, else asm
    order = ['hin', 'ben', 'asm']
    ordered = {lang: cached[lang] for lang in order if lang in cached}
    for lang in cached:
        if lang not in ordered:
            ordered[lang] = cached[lang]
    logger.info(f"Using {len(ordered)} cached language files: {list(ordered.keys())}")

    all_passages = []
    seen_texts = set()

    for lang, path in ordered.items():
        try:
            passages = load_parquet_file(path, lang, max_rows=max_passages_per_lang)
            added = 0
            for p in passages:
                text_key = p['text'][:200]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_passages.append(p)
                    added += 1
            logger.info(f"  {lang}: added {added} new passages (deduplicated)")
        except Exception as e:
            logger.error(f"  Failed to load {lang}: {e}")

    lang_counts = {}
    for p in all_passages:
        l = p.get('language', 'unknown')
        lang_counts[l] = lang_counts.get(l, 0) + 1
    logger.info(f"Total: {len(all_passages)} passages, languages: {lang_counts}")
    return all_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_passages_per_lang', type=int, default=150000)
    parser.add_argument('--output', type=str, default='data/processed_passages.jsonl')
    args = parser.parse_args()

    corpus = load_corpus(args.max_passages_per_lang)
    if not corpus:
        logger.error("No passages loaded!")
        return

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for p in corpus:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(corpus)} passages to {args.output}")


if __name__ == '__main__':
    main()
