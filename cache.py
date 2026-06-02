import collections
import threading
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger("radiance.cache")

_VIEWER_CACHE_MAX = 8
_VIEWER_CACHE: collections.OrderedDict = collections.OrderedDict()
_VIEWER_CACHE_LOCK = threading.Lock()


def _viewer_cache_set(key: str, value: torch.Tensor) -> None:
    """Thread-safe LRU insert into the viewer cache."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
        _VIEWER_CACHE[key] = value
        while len(_VIEWER_CACHE) > _VIEWER_CACHE_MAX:
            evicted_key, evicted_val = _VIEWER_CACHE.popitem(last=False)
            logger.debug(f"[Radiance] Cache evicted node {evicted_key} (capacity {_VIEWER_CACHE_MAX})")
            del evicted_val  # help GC release tensor


def _viewer_cache_get(key: str) -> Optional[torch.Tensor]:
    """Thread-safe LRU lookup."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
            return _VIEWER_CACHE[key]
        return None


# Stores active export progress: {instance_id: {current, total, status, message}}
_VIEWER_PROGRESS_MAX = 32
_VIEWER_PROGRESS: collections.OrderedDict = collections.OrderedDict()
_VIEWER_PROGRESS_LOCK = threading.Lock()


def _progress_set(key: str, value: Dict[str, Any]) -> None:
    """Thread-safe LRU insert into the progress store."""
    with _VIEWER_PROGRESS_LOCK:
        if key in _VIEWER_PROGRESS:
            _VIEWER_PROGRESS.move_to_end(key)
        _VIEWER_PROGRESS[key] = value
        while len(_VIEWER_PROGRESS) > _VIEWER_PROGRESS_MAX:
            _VIEWER_PROGRESS.popitem(last=False)


def _progress_get(key: str) -> Dict[str, Any]:
    """Thread-safe lookup; returns idle sentinel when key is absent."""
    with _VIEWER_PROGRESS_LOCK:
        return dict(_VIEWER_PROGRESS.get(key, {
            "current": 0, "total": 100, "status": "idle", "message": "Waiting...",
        }))
