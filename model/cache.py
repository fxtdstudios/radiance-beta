"""LRU model cache — prevents redundant model loading across node invocations."""
from __future__ import annotations

import logging
from collections import OrderedDict

from radiance.config.env import ENV, get_env_int

logger = logging.getLogger("radiance.model.cache")

_DEFAULT_CACHE_SIZE = get_env_int(ENV.RADIANCE_CACHE_SIZE, 2)


class LRUCache:
    """Least-Recently-Used cache with O(1) get/put/evict."""

    def __init__(self, max_size: int = _DEFAULT_CACHE_SIZE):
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, obj) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                evicted, _ = self._cache.popitem(last=False)
                logger.info("Cache evicted: %s", evicted)
        self._cache[key] = obj

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        self._cache.clear()
        logger.info("Model cache cleared")

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def size(self) -> int:
        return len(self._cache)


_unet_cache = LRUCache()
_clip_cache = LRUCache()
_vae_cache = LRUCache()


def get_model_cache(kind: str) -> LRUCache:
    """Return one of the three singleton caches."""
    return {"unet": _unet_cache, "clip": _clip_cache, "vae": _vae_cache}.get(kind, _unet_cache)


def clear_model_caches() -> None:
    _unet_cache.clear()
    _clip_cache.clear()
    _vae_cache.clear()
