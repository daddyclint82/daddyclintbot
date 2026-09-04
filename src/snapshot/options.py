"""Pure option parsing + env configuration for !snapshot.

No I/O, no imports from the rest of the bot. Everything is validated and
clamped at load; a bad env value logs a WARNING and falls back — never raises.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger('snapshot.options')

DETAIL_LEVELS = ('low', 'medium', 'high')
ACCESS_LEVELS = ('owner', 'staff', 'everyone')


def _clamp_int(name: str, raw, default: int, lo: int, hi: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(f"⚠️ {name}={raw!r} invalid; using default {default}")
        return default
    clamped = max(lo, min(value, hi))
    if clamped != value:
        logger.warning(f"⚠️ {name}={value} out of range [{lo},{hi}]; clamped to {clamped}")
    return clamped


def _clamp_float(name: str, raw, default: float, lo: float, hi: float) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(f"⚠️ {name}={raw!r} invalid; using default {default}")
        return default
    clamped = max(lo, min(value, hi))
    if clamped != value:
        logger.warning(f"⚠️ {name}={value} out of range [{lo},{hi}]; clamped to {clamped}")
    return clamped


def _bool(name: str, raw, default: bool) -> bool:
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in ('true', '1', 'yes', 'on'):
        return True
    if val in ('false', '0', 'no', 'off'):
        return False
    logger.warning(f"⚠️ {name}={raw!r} invalid; using default {default}")
    return default


@dataclass(frozen=True)
class SnapshotConfig:
    access: str = 'owner'
    default_hours: int = 6
    default_detail: str = 'medium'
    top_channels: int = 12
    max_msgs_per_channel: int = 60
    concurrency: int = 4
    channel_timeout: float = 8.0
    max_seconds: float = 25.0
    include_bots: bool = False
    min_msgs_for_mood: int = 3
    new_member_days: int = 7
    cache_ttl: float = 300.0
    cache_max_entries: int = 8
    breaker_cooldown: float = 60.0
    breaker_threshold: int = 3
    llm_enabled: bool = True
    llm_model: str = ''           # '' → fall back to the connector's model
    llm_num_predict: int = 220
    llm_timeout: float = 45.0

    @classmethod
    def from_env(cls) -> 'SnapshotConfig':
        g = os.getenv
        access = (g('SNAPSHOT_ACCESS') or 'owner').strip().lower()
        if access not in ACCESS_LEVELS:
            logger.warning(f"⚠️ SNAPSHOT_ACCESS={access!r} invalid; using 'owner'")
            access = 'owner'
        detail = (g('SNAPSHOT_DEFAULT_DETAIL') or 'medium').strip().lower()
        if detail not in DETAIL_LEVELS:
            logger.warning(f"⚠️ SNAPSHOT_DEFAULT_DETAIL={detail!r} invalid; using 'medium'")
            detail = 'medium'
        return cls(
            access=access,
            default_hours=_clamp_int('SNAPSHOT_DEFAULT_HOURS', g('SNAPSHOT_DEFAULT_HOURS'), 6, 1, 168),
            default_detail=detail,
            top_channels=_clamp_int('SNAPSHOT_TOP_CHANNELS', g('SNAPSHOT_TOP_CHANNELS'), 12, 3, 25),
            max_msgs_per_channel=_clamp_int('SNAPSHOT_MAX_MSGS_PER_CHANNEL', g('SNAPSHOT_MAX_MSGS_PER_CHANNEL'), 60, 5, 500),
            concurrency=_clamp_int('SNAPSHOT_CONCURRENCY', g('SNAPSHOT_CONCURRENCY'), 4, 1, 16),
            channel_timeout=_clamp_float('SNAPSHOT_CHANNEL_TIMEOUT', g('SNAPSHOT_CHANNEL_TIMEOUT'), 8.0, 1.0, 60.0),
            max_seconds=_clamp_float('SNAPSHOT_MAX_SECONDS', g('SNAPSHOT_MAX_SECONDS'), 25.0, 5.0, 120.0),
            include_bots=_bool('SNAPSHOT_INCLUDE_BOTS', g('SNAPSHOT_INCLUDE_BOTS'), False),
            min_msgs_for_mood=_clamp_int('SNAPSHOT_MIN_MSGS_FOR_MOOD', g('SNAPSHOT_MIN_MSGS_FOR_MOOD'), 3, 1, 50),
            new_member_days=_clamp_int('SNAPSHOT_NEW_MEMBER_DAYS', g('SNAPSHOT_NEW_MEMBER_DAYS'), 7, 1, 365),
            cache_ttl=_clamp_float('SNAPSHOT_CACHE_TTL', g('SNAPSHOT_CACHE_TTL'), 300.0, 0.0, 3600.0),
            cache_max_entries=_clamp_int('SNAPSHOT_CACHE_MAX_ENTRIES', g('SNAPSHOT_CACHE_MAX_ENTRIES'), 8, 1, 64),
            breaker_cooldown=_clamp_float('SNAPSHOT_BREAKER_COOLDOWN', g('SNAPSHOT_BREAKER_COOLDOWN'), 60.0, 5.0, 3600.0),
            llm_enabled=_bool('SNAPSHOT_LLM_ENABLED', g('SNAPSHOT_LLM_ENABLED'), True),
            llm_model=(g('SNAPSHOT_MODEL') or '').strip(),
            llm_num_predict=_clamp_int('SNAPSHOT_LLM_NUM_PREDICT', g('SNAPSHOT_LLM_NUM_PREDICT'), 220, 50, 2000),
            llm_timeout=_clamp_float('SNAPSHOT_LLM_TIMEOUT', g('SNAPSHOT_LLM_TIMEOUT'), 45.0, 5.0, 300.0),
        )


@dataclass(frozen=True)
class SnapshotOptions:
    hours: int
    detail: str                       # "low" | "medium" | "high"
    channels: Tuple[str, ...]         # explicit channel names/ids; empty = all
    top: int
    fresh: bool                       # bypass cache
    include_bots: bool


def parse_args(raw: str, cfg: SnapshotConfig) -> Tuple[SnapshotOptions, List[str]]:
    """Parse "!snapshot hours:12 detail:high fresh" → (SnapshotOptions, ignored).

    Pure. Out-of-range values are clamped, unknown tokens are collected into
    `ignored` (the footer lists them), never raises.
    """
    hours = cfg.default_hours
    detail = cfg.default_detail
    top = cfg.top_channels
    channels: Tuple[str, ...] = ()
    fresh = False
    ignored: List[str] = []

    tokens = (raw or '').split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        low = token.lower()
        if low == 'fresh':
            fresh = True
        elif low.startswith('hours:'):
            hours = _clamp_int('hours', low.split(':', 1)[1], cfg.default_hours, 1, 168)
        elif low.startswith('detail:'):
            val = low.split(':', 1)[1]
            if val in DETAIL_LEVELS:
                detail = val
            else:
                ignored.append(token)
        elif low.startswith('channels:'):
            # Consume comma-separated values even when the user adds spaces
            # ("channels:general, memes"): keep eating tokens while the value
            # ends with a comma or the next token starts with one.
            val = token.split(':', 1)[1]
            while i + 1 < len(tokens) and (val.endswith(',')
                                           or tokens[i + 1].startswith(',')):
                i += 1
                val += tokens[i]
            channels = tuple(
                c.strip().lstrip('#') for c in val.split(',') if c.strip()
            )
            if not channels:
                ignored.append(token)
        elif low.startswith('top:'):
            top = _clamp_int('top', low.split(':', 1)[1], cfg.top_channels, 3, 25)
        else:
            ignored.append(token)
        i += 1

    return SnapshotOptions(
        hours=hours, detail=detail, channels=channels,
        top=top, fresh=fresh, include_bots=cfg.include_bots,
    ), ignored
