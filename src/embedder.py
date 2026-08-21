"""Embedder module using raw ONNX Runtime + tokenizers for minimum disk footprint."""
import os
import json
import numpy as np
import structlog
from src.config import settings

logger = structlog.get_logger(__name__)

_session = None
_tokenizer = None
_has_token_type_ids = False

ONNX_DIR = os.path.abspath(settings.EMBED_MODEL_DIR)


def _get_ort():
    global _session, _tokenizer, _has_token_type_ids
    if _session is None:
        model_path = os.path.join(ONNX_DIR, "model_int8.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(ONNX_DIR, "model.onnx")
        if os.path.exists(model_path):
            logger.info(f"Loading ONNX embedding model: {os.path.basename(model_path)}")
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

            from tokenizers import Tokenizer
            tok_path = os.path.join(ONNX_DIR, "tokenizer.json")
            if os.path.exists(tok_path):
                _tokenizer = Tokenizer.from_file(tok_path)
            else:
                from tokenizers import Tokenizer
                _tokenizer = Tokenizer.from_pretrained("intfloat/multilingual-e5-small")

            _has_token_type_ids = any(i.name == "token_type_ids" for i in _session.get_inputs())
        else:
            logger.warning("ONNX model not found")
    return _session


def warmup():
    _get_ort()


def _tokenize(texts):
    from tokenizers import Encoding
    encodings = _tokenizer.encode_batch(texts)
    max_len = max(len(e.ids) for e in encodings) if encodings else 0

    input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

    for i, e in enumerate(encodings):
        input_ids[i, :len(e.ids)] = e.ids
        attention_mask[i, :len(e.attention_mask)] = e.attention_mask

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _mean_pool(token_embs, attention_mask):
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = np.sum(token_embs * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def embed_texts(texts, is_query=False, batch_size=64):
    if not texts:
        return []

    if is_query:
        prefixed = [f"query: {t}" for t in texts]
    else:
        prefixed = [f"passage: {t}" for t in texts]

    session = _get_ort()
    if session is not None:
        out_name = "last_hidden_state" if any(o.name == "last_hidden_state" for o in session.get_outputs()) else "token_embeddings"
        all_embeddings = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i:i+batch_size]
            encoded = _tokenize(batch)
            inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            if _has_token_type_ids:
                inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"], dtype=np.int64)

            outputs = session.run([out_name], inputs)[0]
            batch_emb = _mean_pool(outputs, encoded["attention_mask"].astype(np.float32))
            batch_emb = _l2_normalize(batch_emb).astype(np.float32)
            all_embeddings.append(batch_emb)
        result = np.ascontiguousarray(np.vstack(all_embeddings), dtype=np.float32)
        return [emb.tolist() for emb in result]

    return []
