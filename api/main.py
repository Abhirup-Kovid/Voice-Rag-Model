"""FastAPI application with SSE stream TTL cleanup and lazy harness init."""
import time
import uuid
import asyncio
import json
from typing import Optional, Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import settings
from src.schemas import TextRequest, TextResponse
from src.harness import RAGHarness
from src.latency import latency_monitor

start_time = time.time()
harness: Optional[RAGHarness] = None
_harness_ready = False

active_streams: Dict[str, Tuple[asyncio.Queue, float]] = {}


async def cleanup_streams() -> None:
    while True:
        await asyncio.sleep(30)
        now = time.time()
        expired = [sid for sid, (_, ts) in active_streams.items() if now - ts > 60]
        for sid in expired:
            active_streams.pop(sid, None)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global harness, _harness_ready
    harness = RAGHarness.__new__(RAGHarness)
    asyncio.create_task(cleanup_streams())
    yield

app = FastAPI(title="Voice-Enabled RAG Hackathon", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    from src import pinecone_store
    ready = False
    try:
        pinecone_store.get_index()
        ready = True
    except Exception:
        ready = False
    return {
        "status": "ok",
        "index_loaded": ready,
        "model": settings.GROQ_MODEL,
        "harness_ready": _harness_ready,
        "uptime_sec": int(time.time() - start_time),
    }


def _ensure_harness():
    global harness, _harness_ready
    if not _harness_ready:
        harness = RAGHarness()
        _harness_ready = True


@app.post("/api/text", response_model=TextResponse)
async def process_text(req: TextRequest):
    _ensure_harness()
    return harness.process_text(req.query, req.language)


@app.post("/api/voice")
async def process_voice(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    _ensure_harness()
    stream_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    active_streams[stream_id] = (queue, time.time())
    audio_bytes = await file.read()

    async def run_stream():
        try:
            async for event in harness.process_voice_stream(audio_bytes, language):
                await queue.put(event)
        except Exception as e:
            await queue.put({"event": "done", "error": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_stream())
    return {"stream_id": stream_id}


@app.get("/api/stream/{stream_id}")
async def get_stream(stream_id: str, request: Request):
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    queue, created_at = active_streams[stream_id]
    if time.time() - created_at > 60:
        active_streams.pop(stream_id, None)
        raise HTTPException(status_code=404, detail="Stream expired")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                event = await queue.get()
                if event is None:
                    break
                yield {
                    "event": event.get("event", "message"),
                    "data": json.dumps(event),
                }
        finally:
            active_streams.pop(stream_id, None)

    return EventSourceResponse(event_generator())


@app.get("/api/stats")
async def get_stats():
    return latency_monitor.get_stats()
