"""Discord I/O: walk text channels, fetch history, isolate failures.

Only this module touches the Discord API. Message content is scored for
sentiment and discarded immediately — it is never stored or logged.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List

import discord

from .metrics import ChannelSample
from .options import SnapshotConfig, SnapshotOptions

logger = logging.getLogger('snapshot.collector')


def _readable(channel, me) -> bool:
    try:
        perms = channel.permissions_for(me)
        return bool(perms.view_channel and perms.read_message_history)
    except Exception:
        return False


async def _sample_channel(channel, opts: SnapshotOptions, cfg: SnapshotConfig,
                          analyzer, window_start: datetime, burst_cutoff: datetime,
                          sem: asyncio.Semaphore, deadline: float) -> ChannelSample:
    sample = ChannelSample(
        channel_id=channel.id, name=channel.name,
        category=getattr(getattr(channel, 'category', None), 'name', None),
        status='ok',
    )

    if time.monotonic() >= deadline:
        sample.status, sample.reason = 'skipped', 'deadline'
        return sample

    async def _walk() -> None:
        authors = set()
        async for message in channel.history(
            limit=cfg.max_msgs_per_channel, after=window_start, oldest_first=False
        ):
            if message.author.bot and not opts.include_bots:
                continue  # excludes the bot itself — never feed it its own output
            created = message.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            sample.msg_count += 1
            authors.add(message.author.id)
            sample.author_counts[message.author.id] = \
                sample.author_counts.get(message.author.id, 0) + 1
            sample.last_ts = sample.last_ts or created
            sample.first_ts = created
            if created >= burst_cutoff:
                sample.last_sixth_count += 1

            content = (message.content or '').strip()
            if content:
                # Score and discard — content is never retained.
                sample.compounds.append(
                    analyzer.analyze(content)['compound_score']
                )
        sample.unique_authors = len(authors)

    for attempt in (1, 2):
        try:
            async with sem:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    sample.status, sample.reason = 'skipped', 'deadline'
                    return sample
                await asyncio.wait_for(
                    _walk(), timeout=min(cfg.channel_timeout, max(remaining, 0.5))
                )
            return sample
        except asyncio.TimeoutError:
            sample.status, sample.reason = 'skipped', 'timeout'
            return sample
        except asyncio.CancelledError:
            sample.status, sample.reason = 'skipped', 'deadline'
            return sample
        except discord.Forbidden:
            sample.status, sample.reason = 'skipped', 'no_access'
            return sample
        except discord.HTTPException as e:
            if getattr(e, 'status', None) == 429 and attempt == 1:
                # discord.py normally handles 429 buckets; if one still surfaces,
                # retry once after retry_after (capped), then give up gracefully.
                await asyncio.sleep(min(float(getattr(e, 'retry_after', None) or 1.0), 5.0))
                continue
            sample.status, sample.reason = (
                'skipped', 'rate_limited' if getattr(e, 'status', None) == 429 else 'http_error'
            )
            return sample
        except Exception as e:  # noqa: BLE001 — isolation is the whole point
            logger.warning(f"⚠️ snapshot: #{channel.name} failed: {type(e).__name__}")
            sample.status, sample.reason = 'skipped', 'http_error'
            return sample
    return sample


async def collect(guild, opts: SnapshotOptions, cfg: SnapshotConfig,
                  analyzer) -> List[ChannelSample]:
    """Walk every readable text channel (or the requested subset)."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=opts.hours)
    burst_cutoff = now - timedelta(hours=opts.hours / 6.0)
    deadline = time.monotonic() + cfg.max_seconds
    sem = asyncio.Semaphore(cfg.concurrency)

    selected, pre_skipped = [], []
    for channel in guild.text_channels:
        if opts.channels:
            wanted = set(opts.channels)
            if channel.name not in wanted and str(channel.id) not in wanted:
                continue
        if not _readable(channel, guild.me):
            pre_skipped.append(ChannelSample(
                channel_id=channel.id, name=channel.name,
                category=getattr(getattr(channel, 'category', None), 'name', None),
                status='skipped', reason='no_access',
            ))
            continue
        selected.append(channel)

    tasks = [
        _sample_channel(ch, opts, cfg, analyzer, window_start, burst_cutoff, sem, deadline)
        for ch in selected
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    samples: List[ChannelSample] = []
    for ch, result in zip(selected, results):
        if isinstance(result, Exception):
            logger.warning(f"⚠️ snapshot: #{ch.name} gather error: {type(result).__name__}")
            samples.append(ChannelSample(
                channel_id=ch.id, name=ch.name,
                category=getattr(getattr(ch, 'category', None), 'name', None),
                status='skipped', reason='http_error',
            ))
        else:
            samples.append(result)
    samples.extend(pre_skipped)
    return samples
