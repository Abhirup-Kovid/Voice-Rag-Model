"""STT module using Sarvam API."""
import time
import requests
import structlog
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import settings

logger = structlog.get_logger(__name__)

class STTError(Exception):
    pass

def _get_session():
    session = requests.Session()
    # Retry: 3 attempts with exponential backoff (1s, 2s, 4s) on 5xx, 429
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm", language_code: str = "unknown") -> dict:
    """Transcribe audio using Sarvam API. language_code falls back to auto-detect on failure."""
    url = settings.SARVAM_URL
    headers = {
        "api-subscription-key": settings.SARVAM_API_KEY
    }
    files = {
        "file": (filename, audio_bytes, "audio/webm")
    }
    data = {
        "model": "saarika:v2.5",
        "language_code": language_code or "unknown"
    }
    
    session = _get_session()
    
    try:
        logger.info("Calling Sarvam STT API", language_code=data["language_code"])
        response = session.post(url, headers=headers, files=files, data=data, timeout=10.0)
        if response.status_code != 200 and data["language_code"] != "unknown":
            logger.warning(f"STT rejected language '{data['language_code']}' ({response.status_code}), retrying auto-detect")
            data["language_code"] = "unknown"
            response = session.post(url, headers=headers, files=files, data=data, timeout=10.0)
        response.raise_for_status()
        
        result = response.json()
        transcript = result.get("transcript", "")
        lang_code = result.get("language_code", "en")
        
        return {
            "transcript": transcript,
            "language_code": lang_code,
            "requested_language": data["language_code"]
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"STT API failed: {e}")
        raise STTError(f"Speech-to-Text failed: {str(e)}") from e
