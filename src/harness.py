"""Harness module orchestrating the entire RAG pipeline."""
import time
import uuid
import asyncio
import structlog
from typing import AsyncIterator, Optional

from src.schemas import TextResponse, LatencyBreakdown, GuardrailResult, Source
from src.retriever import retrieve, warmup as retriever_warmup
from src.query_cache import query_cache
from src.guardrails import check_unsafe, check_off_topic, check_grounding, check_refusal
from src.stt_sarvam import transcribe_audio, STTError
from src.llm_groq import generate_stream, generate_full
from src.latency import latency_monitor

logger = structlog.get_logger(__name__)

class RAGHarness:
    def __init__(self):
        logger.info("Initializing RAG Harness...")
        retriever_warmup()
        logger.info("RAG Harness ready.")

    def _fallback_response(self, query: str, reason: str, language: str) -> TextResponse:
        return TextResponse(
            answer="I don't have information about that in my knowledge base." if "topic" in reason else "I cannot process this request.",
            sources=[],
            latency=LatencyBreakdown(),
            guardrail=GuardrailResult(
                passed=False, off_topic="topic" in reason, unsafe="unsafe" in reason,
                grounded=True, refused=True, reason=reason
            ),
            query_id=str(uuid.uuid4()),
            cached=False
        )

    def process_text(self, query: str, language: Optional[str] = None) -> TextResponse:
        """Process a text query synchronously (fallback for non-streaming clients or benchmarking)."""
        start_total = time.time()
        query_id = str(uuid.uuid4())
        logger = structlog.get_logger(__name__).bind(query_id=query_id)
        
        try:
            # 1. Cache Check
            cached = query_cache.get(query)
            if cached:
                logger.info("Cache hit")
                return cached
            
            # 2. Guardrail: Unsafe
            start_gr = time.time()
            is_unsafe = check_unsafe(query)
            if is_unsafe:
                gr_ms = int((time.time() - start_gr) * 1000)
                latency_monitor.record("guardrail", gr_ms)
                total_ms = int((time.time() - start_total) * 1000)
                return TextResponse(
                    answer="I cannot process this request.",
                    sources=[],
                    latency=LatencyBreakdown(guardrail_ms=gr_ms, total_ms=total_ms),
                    guardrail=GuardrailResult(passed=False, off_topic=False, unsafe=True, grounded=True, refused=True, reason="Unsafe content detected."),
                    query_id=query_id,
                    cached=False,
                )
            
            # 3. Guardrail: Off-topic
            is_off_topic, sim = check_off_topic(query)
            gr_ms = int((time.time() - start_gr) * 1000)
            latency_monitor.record("guardrail", gr_ms)
            if is_off_topic:
                total_ms = int((time.time() - start_total) * 1000)
                return TextResponse(
                    answer="I don't have information about that in my knowledge base.",
                    sources=[],
                    latency=LatencyBreakdown(guardrail_ms=gr_ms, total_ms=total_ms),
                    guardrail=GuardrailResult(passed=False, off_topic=True, unsafe=False, grounded=True, refused=True, reason=f"Off-topic (sim={sim:.2f})."),
                    query_id=query_id,
                    cached=False,
                )
                
            # 4. Retrieval (Pinecone hybrid)
            start_ret = time.time()
            sources = retrieve(query, language=language)
            ret_ms = int((time.time() - start_ret) * 1000)
            latency_monitor.record("retrieve", ret_ms) # Embedding + faiss + rerank for simplicity

            # 6. LLM Generation with error handling (always call LLM, even with no sources)
            start_llm = time.time()
            context_chunks = [{"text": s.text, "source_id": f"[{i+1}]"} for i, s in enumerate(sources)]
            try:
                llm_result = generate_full(query, context_chunks, language or "en")
            except Exception as e:
                logger.error("LLM generation failed", error=str(e), exc_info=True)
                llm_error_ms = int((time.time() - start_llm) * 1000)
                total_ms = int((time.time() - start_total) * 1000)
                return TextResponse(
                    answer="I cannot process this request due to an error.",
                    sources=[],
                    latency=LatencyBreakdown(retrieve_ms=ret_ms, llm_total_ms=llm_error_ms, total_ms=total_ms),
                    guardrail=GuardrailResult(passed=False, off_topic=False, unsafe=False, grounded=False, refused=True, reason="LLM error."),
                    query_id=query_id,
                    cached=False,
                )
            llm_total_ms = int((time.time() - start_llm) * 1000)
            latency_monitor.record("llm", llm_total_ms)
            
            # 6. Guardrail: Grounding & Refusal
            start_gr2 = time.time()
            is_refused = check_refusal(llm_result["answer"])
            is_grounded, overlap = check_grounding(llm_result["answer"], [s.text for s in sources])
            gr2_ms = int((time.time() - start_gr2) * 1000)
            latency_monitor.record("guardrail", gr2_ms)
            
            passed = not is_refused
            reason = None
            if is_refused: reason = "Question is outside the knowledge base."
            elif not is_grounded: reason = "Answer may not be grounded in retrieved context."
            
            total_ms = int((time.time() - start_total) * 1000)
            
            resp = TextResponse(
                answer=llm_result["answer"],
                sources=sources,
                latency=LatencyBreakdown(
                    retrieve_ms=ret_ms,
                    guardrail_ms=gr_ms + gr2_ms,
                    llm_first_token_ms=llm_result.get("first_token_ms"),
                    llm_total_ms=llm_total_ms,
                    total_ms=total_ms
                ),
                guardrail=GuardrailResult(
                    passed=passed,
                    off_topic=False,
                    unsafe=False,
                    grounded=is_grounded,
                    refused=is_refused,
                    reason=reason
                ),
                query_id=query_id,
                cached=False
            )
            
            if passed:
                query_cache.set(query, resp)

            return resp
            
        except Exception as e:
            logger.error("Failed to process text query", error=str(e), query=query, exc_info=True)
            return self._fallback_response(query, "Internal error.", language)

    async def process_voice_stream(self, audio_bytes: bytes, language: Optional[str] = None) -> AsyncIterator[dict]:
        """Process voice end-to-end and yield SSE events."""
        start_stt = time.time()
        
        try:
            # 1. STT (runs synchronously in a thread)
            stt_result = await asyncio.to_thread(transcribe_audio, audio_bytes, "audio.webm", language or "unknown")
            query = stt_result["transcript"]
            language = stt_result["language_code"]
            stt_ms = int((time.time() - start_stt) * 1000)
            
            yield {"event": "stt", "transcript": query, "language": language, "stt_ms": stt_ms}
            
            if not query:
                yield {"event": "done", "error": "Could not understand audio."}
                return
                
            # From here, we can process similarly to text but stream LLM
            start_rag = time.time()
            
            # Check Cache
            cached_response = query_cache.get(query)
            if cached_response:
                # Mock a stream from cache
                yield {"event": "token", "text": cached_response.answer}
                # Include cached flag in final data
                final_data = cached_response.model_dump()
                final_data["cached"] = True
                yield {"event": "done", "data": final_data}
                return
                
            # Guardrails early
            if check_unsafe(query):
                yield {"event": "token", "text": "I cannot answer this request. (Unsafe)"}
                yield {"event": "done", "error": "Unsafe query"}
                return
                
            is_off, sim = check_off_topic(query)
            if is_off:
                yield {"event": "token", "text": "I don't have information about that in my knowledge base."}
                yield {"event": "done", "error": f"Off-topic (sim={sim:.2f})"}
                return
                
            # Retrieve
            start_ret = time.time()
            sources = await asyncio.to_thread(retrieve, query, language)
            ret_ms = int((time.time() - start_ret) * 1000)
            logger.info(f"voice retrieve_ms={ret_ms} query={query!r}")
            
            if not sources:
                yield {"event": "token", "text": "I don't have information about that in my knowledge base."}
                yield {"event": "done", "error": "No sources"}
                return
                
            # LLM Stream
            context_chunks = [{"text": s.text, "source_id": f"[{i+1}]"} for i, s in enumerate(sources)]
            
            start_llm = time.time()
            first_token_ms = None
            full_answer = ""
            
            # Iterate over the stream generator
            generator = generate_stream(query, context_chunks, language)
            for chunk in generator:
                if chunk["event"] == "first_token_ms":
                    first_token_ms = chunk["value"]
                elif chunk["event"] == "token":
                    full_answer += chunk["text"]
                    yield chunk
                    
            llm_total_ms = int((time.time() - start_llm) * 1000)
            
            # Grounding check
            is_grounded, overlap = check_grounding(full_answer, [s.text for s in sources])
            is_refused = check_refusal(full_answer)
            
            total_rag_ms = int((time.time() - start_rag) * 1000)
            
            final_data = TextResponse(
                answer=full_answer,
                sources=sources,
                latency=LatencyBreakdown(
                    stt_ms=stt_ms,
                    retrieve_ms=ret_ms,
                    llm_first_token_ms=first_token_ms,
                    llm_total_ms=llm_total_ms,
                    total_ms=total_rag_ms
                ),
                guardrail=GuardrailResult(
                    passed=not is_refused and is_grounded,
                    off_topic=False, unsafe=False,
                    grounded=is_grounded, refused=is_refused,
                    reason=None
                ),
                query_id=str(uuid.uuid4()),
                cached=False
            ).model_dump()
            
            yield {"event": "done", "data": final_data}
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"event": "done", "error": str(e)}
