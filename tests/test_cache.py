"""TTL, LRU eviction, breaker trip/half-open. Pure (tiny sleeps only)."""

import asyncio

from snapshot.cache import CircuitBreaker, TTLCache


def test_hit_returns_value_and_age():
    cache = TTLCache(ttl=60, max_entries=4)
    cache.put('k', 'v')
    hit = cache.get('k')
    assert hit is not None
    value, age = hit
    assert value == 'v' and age >= 0


def test_miss_returns_none():
    assert TTLCache(ttl=60, max_entries=4).get('nope') is None


async def test_ttl_expiry():
    cache = TTLCache(ttl=0.05, max_entries=4)
    cache.put('k', 'v')
    await asyncio.sleep(0.08)
    assert cache.get('k') is None


def test_lru_eviction_bounds_memory():
    cache = TTLCache(ttl=60, max_entries=3)
    for i in range(10):
        cache.put(f'k{i}', i)
    assert len(cache) == 3
    assert cache.get('k9') is not None       # newest survives
    assert cache.get('k6') is None           # oldest evicted (k7..k9 remain)


def test_lru_hit_refreshes_recency():
    cache = TTLCache(ttl=60, max_entries=2)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.get('a')                            # a now most-recent
    cache.put('c', 3)                         # evicts b, not a
    assert cache.get('a') is not None
    assert cache.get('b') is None


def test_breaker_trips_after_threshold():
    br = CircuitBreaker(threshold=3, cooldown=60)
    assert br.allow() is True
    br.record_failure()
    br.record_failure()
    assert br.allow() is True                 # still closed at 2
    br.record_failure()
    assert br.allow() is False                # open at 3
    assert br.is_open is True
    assert br.cooldown_remaining() > 0


async def test_breaker_half_open_after_cooldown():
    br = CircuitBreaker(threshold=2, cooldown=0.05)
    br.record_failure()
    br.record_failure()
    assert br.allow() is False
    await asyncio.sleep(0.08)
    assert br.allow() is True                 # half-open


def test_breaker_success_resets():
    br = CircuitBreaker(threshold=2, cooldown=60)
    br.record_failure()
    br.record_success()
    assert br.failures == 0 and br.opened_at is None
