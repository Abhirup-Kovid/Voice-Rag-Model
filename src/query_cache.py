"""LRU Query Cache."""
import re
import threading
from cachetools import LRUCache
from src.config import settings
from src.schemas import TextResponse

class QueryCache:
    def __init__(self, maxsize: int):
        self.cache = LRUCache(maxsize=maxsize)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
    def _normalize(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        return query
        
    def get(self, query: str) -> TextResponse | None:
        key = self._normalize(query)
        with self.lock:
            if key in self.cache:
                self.hits += 1
                resp = self.cache[key]
                # Create a copy so we can modify 'cached' property safely
                resp_copy = resp.model_copy()
                resp_copy.cached = True
                return resp_copy
            else:
                self.misses += 1
                return None
                
    def set(self, query: str, response: TextResponse):
        key = self._normalize(query)
        with self.lock:
            # Store with cached=False so it reflects the original state
            self.cache[key] = response
            
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

query_cache = QueryCache(maxsize=settings.CACHE_SIZE)
