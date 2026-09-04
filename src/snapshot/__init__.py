"""!snapshot — on-demand, pull-only server awareness.

Public surface: run_snapshot() and SnapshotOptions. No listeners, no loops,
no schedulers — all work happens inside the command invocation, and results
live at most in a short-TTL in-memory cache.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

from . import collector, metrics, render, summarizer
from .cache import CircuitBreaker, TTLCache
from .options import SnapshotConfig, SnapshotOptions, parse_args
from .render import RenderedOutcome

__all__ = ['run_snapshot', 'SnapshotOptions', 'SnapshotConfig', 'RenderedOutcome']

logger = logging.getLogger('snapshot')

_config: Optional[SnapshotConfig] = None
_cache: Optional[TTLCache] = None
_breaker: Optional[CircuitBreaker] = None
_inflight: Dict[Tuple, asyncio.Task] = {}


def get_config() -> SnapshotConfig:
    global _config
    if _config is None:
        _config = SnapshotConfig.from_env()
    return _config


def _get_cache(cfg: SnapshotConfig) -> TTLCache:
    global _cache
    if _cache is None or _cache.ttl != cfg.cache_ttl \
            or _cache.max_entries != cfg.cache_max_entries:
        _cache = TTLCache(cfg.cache_ttl, cfg.cache_max_entries)
    return _cache


def _get_breaker(cfg: SnapshotConfig) -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker(cfg.breaker_threshold, cfg.breaker_cooldown)
    return _breaker


def _allowed(cfg: SnapshotConfig, author, is_owner: bool) -> bool:
    if is_owner:
        return True
    if cfg.access == 'everyone':
        return True
    if cfg.access == 'staff':
        perms = getattr(author, 'guild_permissions', None)
        return bool(getattr(perms, 'manage_messages', False))
    return False  # owner-only default


async def _execute(*, guild, is_owner: bool, opts: SnapshotOptions,
                   cfg: SnapshotConfig, ignored, engine) -> RenderedOutcome:
    """collect → metrics → summarize → render. Always returns an outcome."""
    started = time.monotonic()
    cache = _get_cache(cfg)
    cache_key = (guild.id, opts.hours, opts.detail if is_owner else 'low',
                 opts.channels, opts.include_bots, is_owner)

    if not opts.fresh:
        hit = cache.get(cache_key)
        if hit is not None:
            (snapshot, summary), age = hit
            logger.info(f"📸 snapshot cache hit (age {int(age)}s) guild={guild.id}")
            return render.build(snapshot, summary, opts=opts, cfg=cfg, guild=guild,
                                is_owner=is_owner, cache_age=age, ignored=ignored,
                                model_name=cfg.llm_model or engine.llm.model)

    samples = await collector.collect(guild, opts, cfg, engine.analyzer)
    elapsed = time.monotonic() - started

    snapshot = metrics.compute(samples, opts, cfg, guild=guild,
                               guild_name=guild.name, elapsed_s=elapsed,
                               for_owner=is_owner)
    summary = await summarizer.summarize(snapshot, opts, cfg, engine.llm)

    # One structured INFO line — counts and timings only, never content.
    logger.info(
        f"📸 snapshot guild={guild.id} window={opts.hours}h "
        f"scanned={snapshot.scanned} skipped={snapshot.skipped} "
        f"msgs={snapshot.total_msgs} voices={snapshot.total_unique_authors} "
        f"elapsed={snapshot.elapsed_s:.1f}s llm={'ok' if summary else 'fallback'} "
        f"owner={is_owner}"
    )

    if snapshot.scanned == 0 and snapshot.skipped > 0:
        # Zero readable channels: clear actionable reply, still no exception.
        return RenderedOutcome(
            embed=None,
            plain_text="📸 0 channels readable — my role needs **View Channel** + "
                       "**Read Message History** on at least one channel. "
                       "(check role permissions, then try again)",
        )

    cache.put(cache_key, (snapshot, summary))
    return render.build(snapshot, summary, opts=opts, cfg=cfg, guild=guild,
                        is_owner=is_owner, cache_age=None, ignored=ignored,
                        model_name=cfg.llm_model or engine.llm.model)


async def run_snapshot(*, guild, author, is_owner: bool, raw_args: str,
                       engine) -> RenderedOutcome:
    """Entry point for the !snapshot command. Never raises."""
    cfg = get_config()
    opts, ignored = parse_args(raw_args, cfg)

    if not _allowed(cfg, author, is_owner):
        logger.info(f"📸 snapshot refused for user {getattr(author, 'id', '?')} "
                    f"(access={cfg.access})")
        return RenderedOutcome(embed=None,
                               plain_text="that one's not in your toolkit, sorry 💀")

    breaker = _get_breaker(cfg)
    if not breaker.allow():
        wait = int(breaker.cooldown_remaining())
        return RenderedOutcome(embed=None,
                               plain_text=f"snapshot's cooling down — try again in {wait}s 🧊")

    flight_key = (guild.id, opts.hours, opts.detail if is_owner else 'low',
                  opts.channels, opts.include_bots, is_owner)
    task = _inflight.get(flight_key)
    if task is not None and not task.done():
        # Identical snapshot already running — share its result with this caller.
        try:
            return await asyncio.shield(task)
        except Exception:
            pass  # fall through and run fresh
    elif any(not t.done() for t in _inflight.values()):
        # Different args already in flight — tell the caller, don't stack runs.
        return RenderedOutcome(embed=None, plain_text="one's already cooking 🔥")

    task = asyncio.create_task(_execute(guild=guild, is_owner=is_owner, opts=opts,
                                        cfg=cfg, ignored=ignored, engine=engine))
    _inflight[flight_key] = task
    try:
        outcome = await asyncio.shield(task)
        breaker.record_success()
        return outcome
    except Exception as e:  # noqa: BLE001 — catch-all: log (no content), apologize, trip
        logger.error(f"❌ snapshot failed: {type(e).__name__}: {e}", exc_info=True)
        breaker.record_failure()
        return RenderedOutcome(
            embed=None,
            plain_text="snapshot glitched on me — try again in a sec 💀",
        )
    finally:
        _inflight.pop(flight_key, None)
