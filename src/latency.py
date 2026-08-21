"""Latency tracking module."""
import time
import threading
from functools import wraps
import numpy as np
from typing import Dict, List
import structlog

logger = structlog.get_logger(__name__)

class LatencyMonitor:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.data: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
        
    def record(self, stage: str, duration_ms: float):
        with self.lock:
            if stage not in self.data:
                self.data[stage] = []
            self.data[stage].append(duration_ms)
            if len(self.data[stage]) > self.capacity:
                self.data[stage] = self.data[stage][-self.capacity:]
                
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        with self.lock:
            for stage, values in self.data.items():
                if not values:
                    continue
                arr = np.array(values)
                stats[stage] = {
                    "p50": float(np.percentile(arr, 50)),
                    "p70": float(np.percentile(arr, 70)),
                    "p100": float(np.max(arr)),
                    "samples": len(values)
                }
        return stats

latency_monitor = LatencyMonitor()

# measure_stage decorator removed (unused)
