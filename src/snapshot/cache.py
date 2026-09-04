"""TTL cache (LRU) + circuit breaker for !snapshot. Pure, no I/O."""

import time
from collections import OrderedDict
from typing import Any, Optional, Tuple


class TTLCache:
    """In-memory TTL cache with LRU eviction. Bounded by max_entries."""

    def __init__(self, ttl: float = 300.0, max_entries: int = 8):
        self.ttl = ttl
        self.max_entries = max_entries
        self._store: "OrderedDict[Any, Tuple[Any, float]]" = OrderedDict()

    def get(self, key) -> Optional[Tuple[Any, float]]:
        """Return (value, age_seconds) if present and fresh, else None."""
        item = self._store.get(key)
        if item is None:
            return None
        value, stored_at = item
        age = time.monotonic() - stored_at
        if age > self.ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value, age

    def put(self, key, value) -> None:
        self._store[key] = (value, time.monotonic())
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class CircuitBreaker:
    """Open after `threshold` consecutive failures; half-open after cooldown."""

    def __init__(self, threshold: int = 3, cooldown: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at: Optional[float] = None

    def allow(self) -> bool:
        """Closed → True. Open → False until cooldown elapses (then half-open)."""
        if self.opened_at is None:
            return True
        return (time.monotonic() - self.opened_at) >= self.cooldown

    def cooldown_remaining(self) -> float:
        if self.opened_at is None:
            return 0.0
        return max(0.0, self.cooldown - (time.monotonic() - self.opened_at))

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None and not self.allow()

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()
