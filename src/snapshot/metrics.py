"""Pure metrics: ChannelSample list → ServerSnapshot. No I/O, no discord imports."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from agent import DaddyClintBot  # for the shared _mood_label (consistency with !vibe)

from .options import SnapshotConfig, SnapshotOptions


@dataclass
class ChannelSample:
    channel_id: int
    name: str
    category: Optional[str]
    status: str                     # "ok" | "skipped"
    reason: Optional[str] = None    # no_access | rate_limited | timeout | http_error | deadline
    msg_count: int = 0
    unique_authors: int = 0
    compounds: List[float] = field(default_factory=list)
    author_counts: Dict[int, int] = field(default_factory=dict)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    last_sixth_count: int = 0       # msgs in the final 1/6 of the window (burst detection)


@dataclass
class ChannelMetrics:
    name: str
    msg_count: int
    unique_authors: int
    avg_compound: Optional[float]
    mood: str                       # via engine._mood_label, or "n/a"
    activity_tier: str              # "hot" | "warm" | "low" | "quiet" | "skipped"
    burst: bool
    top_authors: List[Tuple[int, int]] = field(default_factory=list)  # (author_id, count), top 5


@dataclass
class ServerSnapshot:
    guild_name: str
    window_hours: int
    taken_at: datetime
    channels: List[ChannelMetrics]          # sorted by msg_count desc, skipped last
    scanned: int
    skipped: int
    skipped_reasons: Dict[str, int]
    total_msgs: int
    total_unique_authors: int
    highlights: List[str]                   # rule-based, ≤ 6
    partial: bool
    elapsed_s: float
    overall_mood: str = "n/a"


def _tier(msg_count: int, window_hours: int) -> str:
    """Thresholds scale linearly: hot ≥ 30 msgs per 6h-equivalent, warm ≥ 10."""
    scale = window_hours / 6.0
    if msg_count >= 30 * scale:
        return 'hot'
    if msg_count >= 10 * scale:
        return 'warm'
    if msg_count >= 1:
        return 'low'
    return 'quiet'


def _clip(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[:limit - 1] + '…'


def compute(samples: List[ChannelSample], opts: SnapshotOptions,
            cfg: SnapshotConfig, *, guild=None, guild_name: str = 'server',
            elapsed_s: float = 0.0, for_owner: bool = False,
            now: Optional[datetime] = None) -> ServerSnapshot:
    now = now or datetime.now(timezone.utc)

    channels: List[ChannelMetrics] = []
    skipped_reasons: Dict[str, int] = {}
    all_compounds: List[float] = []
    all_authors: set = set()

    for s in samples:
        if s.status != 'ok':
            skipped_reasons[s.reason or 'unknown'] = skipped_reasons.get(s.reason or 'unknown', 0) + 1
            channels.append(ChannelMetrics(
                name=s.name, msg_count=0, unique_authors=0, avg_compound=None,
                mood='n/a', activity_tier='skipped', burst=False,
            ))
            continue

        avg = (sum(s.compounds) / len(s.compounds)
               if len(s.compounds) >= cfg.min_msgs_for_mood else None)
        mood = DaddyClintBot._mood_label(avg) if avg is not None else 'n/a'
        burst = s.last_sixth_count >= 0.4 * s.msg_count and s.msg_count >= 6
        top = sorted(s.author_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        channels.append(ChannelMetrics(
            name=s.name, msg_count=s.msg_count, unique_authors=s.unique_authors,
            avg_compound=avg, mood=mood, activity_tier=_tier(s.msg_count, opts.hours),
            burst=burst, top_authors=top,
        ))
        all_compounds.extend(s.compounds)
        all_authors.update(s.author_counts.keys())

    # busiest first, skipped at the end
    channels.sort(key=lambda c: (c.activity_tier == 'skipped', -c.msg_count, c.name))

    scanned = sum(1 for c in channels if c.activity_tier != 'skipped')
    skipped = len(channels) - scanned
    total_msgs = sum(c.msg_count for c in channels)
    overall = (DaddyClintBot._mood_label(sum(all_compounds) / len(all_compounds))
               if all_compounds else 'n/a')

    highlights = _highlights(channels, skipped_reasons, samples, cfg,
                             guild=guild, for_owner=for_owner, now=now)

    return ServerSnapshot(
        guild_name=guild_name, window_hours=opts.hours, taken_at=now,
        channels=channels, scanned=scanned, skipped=skipped,
        skipped_reasons=skipped_reasons, total_msgs=total_msgs,
        total_unique_authors=len(all_authors), highlights=highlights,
        partial=skipped > 0, elapsed_s=elapsed_s, overall_mood=overall,
    )


def _highlights(channels: List[ChannelMetrics], skipped_reasons: Dict[str, int],
                samples: List[ChannelSample], cfg: SnapshotConfig, *,
                guild, for_owner: bool, now: datetime) -> List[str]:
    out: List[str] = []
    ok = [c for c in channels if c.activity_tier != 'skipped']

    # 1. hottest channel
    if ok and ok[0].msg_count > 0:
        out.append(_clip(f"#{ok[0].name} — {ok[0].msg_count} msgs, "
                         f"{ok[0].unique_authors} voices"))

    # 2. bursts
    for c in ok:
        if c.burst:
            out.append(_clip(f"📈 #{c.name} heating up in the last hour"))

    # 3. sentiment extremes
    for c in ok:
        if c.avg_compound is None or c.msg_count < 5:
            continue
        if c.avg_compound >= 0.5:
            out.append(_clip(f"🔥 #{c.name} is hyped"))
        elif c.avg_compound <= -0.3:
            out.append(_clip(f"⚠️ #{c.name} is tense"))

    # 4. silent share
    quiet = sum(1 for c in ok if c.activity_tier == 'quiet')
    if quiet and len(channels):
        out.append(_clip(f"⚫ {quiet} of {len(channels)} channels quiet"))

    # 5. new voices (joined recently and posted in-window)
    if guild is not None:
        new_ids, new_names = [], []
        seen = set()
        for s in samples:
            if s.status != 'ok':
                continue
            for author_id in s.author_counts:
                if author_id in seen:
                    continue
                seen.add(author_id)
                member = guild.get_member(author_id)
                joined = getattr(member, 'joined_at', None) if member else None
                if joined and (now - joined).days <= cfg.new_member_days:
                    new_ids.append(author_id)
                    new_names.append(getattr(member, 'display_name', f'user-{author_id}'))
        if new_ids:
            if for_owner:
                out.append(_clip("🌱 new voices: " + ", ".join(new_names[:5])))
            else:
                out.append(_clip(f"🌱 {len(new_ids)} new voices in the mix"))

    # 6. skips
    total_skipped = sum(skipped_reasons.values())
    if total_skipped:
        reasons = ", ".join(f"{r.replace('_', ' ')}" for r in skipped_reasons)
        out.append(_clip(f"🔒 {total_skipped} channels skipped ({reasons})"))

    return out[:6]
