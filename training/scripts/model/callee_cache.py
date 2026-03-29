# training/scripts/model/callee_cache.py
#
# Redis-backed cache for CodeBERT callee embeddings.
# Falls back to in-memory dict when Redis is unavailable (training mode).
#
# Used by CalleeSummarizer for Config E ablation only.
# Key: "callee:" + sha256(callee_source_code)
# Value: 768-d float32 tensor (JSON-serialized list for Redis, direct tensor for memory)

import hashlib
import json
import logging

import torch

logger = logging.getLogger(__name__)


class CalleeCache:
    """
    Cache for CodeBERT callee embeddings.

    Redis backend with automatic in-memory fallback.
    Permanent TTL (callee embeddings don't change during training).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        use_memory_fallback: bool = True,
    ):
        self._memory: dict[str, torch.Tensor] = {}
        self._redis = None
        self._use_redis = False

        try:
            import redis as redis_lib
            r = redis_lib.Redis.from_url(
                redis_url, socket_timeout=2, socket_connect_timeout=2,
            )
            r.ping()
            self._redis = r
            self._use_redis = True
            logger.info("CalleeCache: Redis connected at %s", redis_url)
        except Exception as e:
            if use_memory_fallback:
                logger.warning("CalleeCache: Redis unavailable (%s), using memory cache", e)
            else:
                raise

    @staticmethod
    def _key(code: str) -> str:
        """Generate cache key: 'callee:' + sha256(code)."""
        return "callee:" + hashlib.sha256(code.encode("utf-8")).hexdigest()

    def get(self, code: str) -> torch.Tensor | None:
        """Look up cached embedding. Returns 768-d tensor or None on miss."""
        k = self._key(code)

        if self._use_redis:
            raw = self._redis.get(k)
            if raw is not None:
                return torch.tensor(json.loads(raw), dtype=torch.float32)
        else:
            if k in self._memory:
                return self._memory[k].clone()

        return None

    def set(self, code: str, embed: torch.Tensor) -> None:
        """Store embedding in cache."""
        k = self._key(code)

        if self._use_redis:
            self._redis.set(k, json.dumps(embed.tolist()))
        else:
            self._memory[k] = embed.detach().cpu().clone()

    def size(self) -> int:
        """Number of cached entries."""
        if self._use_redis:
            return self._redis.dbsize()
        return len(self._memory)

    @property
    def is_redis(self) -> bool:
        """Whether Redis backend is active."""
        return self._use_redis
