"""Render a ServerSnapshot to one Discord embed (+ plain-text fallback).

Hard-enforces Discord embed limits (description 4,096 · field 1,024 ·
total 6,000 · 25 fields). Non-owners always get the count-only view and
never see member names.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

import discord

from .metrics import ServerSnapshot
from .options import SnapshotConfig, SnapshotOptions

logger = logging.getLogger('snapshot.render')

TIER_EMOJI = {'hot': '🟢', 'warm': '🟡', 'low': '⚪', 'quiet': '⚫', 'skipped': '🔒'}
LOCAL_TZ = ZoneInfo('America/Chicago')

EMBED_TOTAL_LIMIT = 6000
FIELD_LIMIT = 1024
DESCRIPTION_LIMIT = 4096


@dataclass
class RenderedOutcome:
    embed: Optional[discord.Embed]
    plain_text: str


def _channel_lines(snapshot: ServerSnapshot, opts: SnapshotOptions,
                   with_mood: bool) -> List[str]:
    lines = []
    shown = 0
    for c in snapshot.channels:
        if c.activity_tier == 'skipped':
            continue
        if shown >= opts.top:
            break
        emoji = TIER_EMOJI[c.activity_tier]
        if with_mood:
            lines.append(f"{emoji} #{c.name:<18} {c.msg_count:>4} msgs · "
                         f"{c.unique_authors:>2} voices · {c.mood}")
        else:
            lines.append(f"{emoji} #{c.name:<18} {c.msg_count:>4} msgs · "
                         f"{c.unique_authors:>2} voices")
        shown += 1
    remaining = sum(1 for c in snapshot.channels
                    if c.activity_tier == 'quiet') - \
        sum(1 for c in snapshot.channels[:opts.top] if c.activity_tier == 'quiet')
    if remaining > 0:
        lines.append(f"+ {remaining} more quiet")
    return lines


def _fallback_read(snapshot: ServerSnapshot) -> str:
    hot = sum(1 for c in snapshot.channels if c.activity_tier == 'hot')
    quiet = sum(1 for c in snapshot.channels if c.activity_tier == 'quiet')
    return (f"{hot} channels hot, {quiet} quiet; mood {snapshot.overall_mood}.\n"
            f"(AI read unavailable)")


def build(snapshot: ServerSnapshot, summary: Optional[str], *,
          opts: SnapshotOptions, cfg: SnapshotConfig, guild,
          is_owner: bool, cache_age: Optional[float] = None,
          ignored: Optional[List[str]] = None,
          model_name: str = '') -> RenderedOutcome:
    """Build the embed (and its plain-text twin). Non-owner ⇒ count-only."""
    # Non-owners always get the count-only view regardless of detail: (§5.2)
    effective_detail = opts.detail if is_owner else 'low'
    with_mood = effective_detail in ('medium', 'high')

    local_time = snapshot.taken_at.astimezone(LOCAL_TZ).strftime('%b %d, %I:%M %p %Z')
    title = f"📸 NSZ Snapshot — {local_time}"

    description = (f"Window: last {snapshot.window_hours}h · "
                   f"Channels: {snapshot.scanned} scanned, {snapshot.skipped} skipped · "
                   f"Messages: {snapshot.total_msgs} · "
                   f"Voices: {snapshot.total_unique_authors}")
    if snapshot.partial:
        reason = ('deadline hit' if snapshot.skipped_reasons.get('deadline')
                  else 'some channels skipped')
        description += f"\n⚠️ Partial — {reason}"
    description = description[:DESCRIPTION_LIMIT]

    embed = discord.Embed(title=title[:256], description=description,
                          color=discord.Color.teal())

    channel_lines = _channel_lines(snapshot, opts, with_mood)
    embed.add_field(
        name="Channels",
        value=("```\n" + "\n".join(channel_lines) + "\n```")[:FIELD_LIMIT] or "none",
        inline=False,
    )

    if effective_detail in ('medium', 'high') and snapshot.highlights:
        embed.add_field(
            name="Highlights",
            value="\n".join(snapshot.highlights)[:FIELD_LIMIT],
            inline=False,
        )

    read_text = summary or _fallback_read(snapshot)
    embed.add_field(name="Read", value=read_text[:FIELD_LIMIT], inline=False)

    if is_owner and effective_detail == 'high':
        voices = []
        for c in snapshot.channels:
            for author_id, count in c.top_authors:
                member = guild.get_member(author_id) if guild else None
                name = getattr(member, 'display_name', None) or f"user-{author_id}"
                voices.append((name, count))
        # merge per-author across channels, top 5 overall
        merged = {}
        for name, count in voices:
            merged[name] = merged.get(name, 0) + count
        top = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:5]
        if top:
            embed.add_field(name="Top voices",
                            value=" · ".join(f"@{n} ×{c}" for n, c in top)[:FIELD_LIMIT],
                            inline=False)

    footer_parts = [f"took {snapshot.elapsed_s:.1f}s"]
    if cache_age is not None:
        footer_parts.append(f"cache {int(cache_age)}s")
        if is_owner:
            footer_parts.append('add "fresh" to refetch')
    footer_parts.append(f"model {model_name or 'n/a'}")
    if ignored:
        footer_parts.append("ignored: " + ", ".join(ignored[:5]))
    embed.set_footer(text=" · ".join(footer_parts)[:2048])

    _enforce_limits(embed)
    return RenderedOutcome(embed=embed, plain_text=_plain(snapshot, summary, opts,
                                                          effective_detail, cache_age,
                                                          ignored, model_name))


def _enforce_limits(embed: discord.Embed) -> None:
    """Trim in order: Top voices → Highlights → Channels tail, until legal."""
    def total_size() -> int:
        size = len(embed.title or '') + len(embed.description or '')
        footer = embed.footer.text if embed.footer else None
        size += len(footer or '')
        for f in embed.fields:
            size += len(f.name) + len(f.value)
        return size

    # Per-field cap first
    for f in embed.fields:
        if len(f.value) > FIELD_LIMIT:
            f.value = f.value[:FIELD_LIMIT - 1] + '…'

    def drop_field(name: str) -> bool:
        for i, f in enumerate(embed.fields):
            if f.name == name:
                embed.remove_field(i)
                return True
        return False

    def shrink_channels() -> bool:
        for f in embed.fields:
            if f.name != 'Channels':
                continue
            lines = f.value.splitlines()
            if len(lines) <= 4:  # nothing left to shrink — drop the field
                return drop_field('Channels')
            trimmed = lines[:-2]
            if trimmed and trimmed[-1].strip() != '```':
                trimmed.append('```')
            f.value = "\n".join(trimmed)
            return True
        return False

    for action in (lambda: drop_field('Top voices'),
                   lambda: drop_field('Highlights'),
                   shrink_channels):
        while total_size() > EMBED_TOTAL_LIMIT:
            if not action():
                break


def _plain(snapshot: ServerSnapshot, summary: Optional[str], opts: SnapshotOptions,
           effective_detail: str, cache_age: Optional[float],
           ignored: Optional[List[str]], model_name: str) -> str:
    lines = [f"📸 NSZ Snapshot — {snapshot.taken_at.astimezone(LOCAL_TZ).strftime('%b %d, %I:%M %p %Z')}"]
    lines.append(f"Window: last {snapshot.window_hours}h · "
                 f"{snapshot.scanned} scanned, {snapshot.skipped} skipped · "
                 f"{snapshot.total_msgs} msgs · {snapshot.total_unique_authors} voices")
    if snapshot.partial:
        lines.append("⚠️ Partial results")
    with_mood = effective_detail in ('medium', 'high')
    for line in _channel_lines(snapshot, opts, with_mood):
        lines.append(line)
    if effective_detail in ('medium', 'high') and snapshot.highlights:
        lines.append("Highlights: " + " | ".join(snapshot.highlights))
    lines.append("Read: " + (summary or _fallback_read(snapshot)).replace("\n", " / "))
    if cache_age is not None:
        lines.append(f"(cached {int(cache_age)}s)")
    if ignored:
        lines.append("ignored: " + ", ".join(ignored[:5]))
    return "\n".join(lines)
