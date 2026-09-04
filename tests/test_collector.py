"""Failure isolation: Forbidden/429/timeout/deadline, bots skipped, content dropped."""

import asyncio
from dataclasses import replace

import discord

from snapshot.collector import collect
from tests.conftest import FakeChannel, FakeMessage, FakeResponse, make_channel


def by_name(samples):
    return {s.name: s for s in samples}


async def test_happy_path_counts(guild, opts, cfg, analyzer):
    samples = by_name(await collect(guild, opts, cfg, analyzer))
    gen = samples['general']
    assert gen.status == 'ok'
    # 6 human messages (bot skipped by default, empty content still counted)
    assert gen.msg_count == 6
    assert gen.unique_authors == 3                    # alice, bob, carol
    assert len(gen.compounds) == 5                    # empty message scored 0 sentiment
    assert samples['quiet-corner'].msg_count == 0
    assert samples['quiet-corner'].status == 'ok'


async def test_no_access_without_api_call(guild, opts, cfg, analyzer):
    samples = by_name(await collect(guild, opts, cfg, analyzer))
    locked = samples['locked-ops']
    assert locked.status == 'skipped' and locked.reason == 'no_access'
    locked_fake = next(c for c in guild.text_channels if c.name == 'locked-ops')
    assert locked_fake.history_calls == 0             # pre-checked, never called


async def test_bots_excluded_by_default_and_counted_when_opted_in(guild, opts, cfg, analyzer):
    samples = by_name(await collect(guild, opts, cfg, analyzer))
    assert samples['general'].msg_count == 6

    opts_bots = replace(opts, include_bots=True)
    samples = by_name(await collect(guild, opts_bots, cfg, analyzer))
    assert samples['general'].msg_count == 7          # botfriend counted now


async def test_content_never_retained(guild, opts, cfg, analyzer):
    samples = await collect(guild, opts, cfg, analyzer)
    for s in samples:
        assert not hasattr(s, 'content')
        assert all(isinstance(c, float) for c in s.compounds)


async def test_forbidden_isolated(guild, opts, cfg, analyzer):
    forbidden = discord.Forbidden(FakeResponse(403, 'Forbidden'), 'missing access')
    bad = FakeChannel('broken', 99, raise_exc=forbidden)
    guild.text_channels.append(bad)
    samples = by_name(await collect(guild, opts, cfg, analyzer))
    assert samples['broken'].status == 'skipped'
    assert samples['broken'].reason == 'no_access'
    assert samples['general'].status == 'ok'          # everyone else unaffected


async def test_http_error_isolated(guild, opts, cfg, analyzer):
    err = discord.HTTPException(FakeResponse(500, 'Server Error'), 'boom')
    bad = FakeChannel('flaky', 98, raise_exc=err)
    guild.text_channels.append(bad)
    samples = by_name(await collect(guild, opts, cfg, analyzer))
    assert samples['flaky'].reason == 'http_error'
    assert samples['general'].status == 'ok'


async def test_429_retried_once_then_ok(guild, opts, cfg, analyzer):
    err = discord.HTTPException(FakeResponse(429, 'Too Many'), 'rate limited')
    err.status = 429
    err.retry_after = 0.01
    flaky = make_channel('ratelimited', 97,
                         [(guild._members[101], 'hello there friend', 5)],
                         raise_exc=err, raise_once=True)
    guild.text_channels.append(flaky)
    samples = by_name(await collect(guild, opts, cfg, analyzer))
    assert samples['ratelimited'].status == 'ok'      # recovered on the retry
    assert samples['ratelimited'].msg_count == 1
    assert flaky.history_calls == 2                   # exactly one retry


async def test_per_channel_timeout(guild, opts, cfg, analyzer):
    slow = make_channel('slowpoke', 96,
                        [(guild._members[101], 'hi', 5)],
                        per_message_sleep=0.3)
    guild.text_channels.append(slow)
    fast_cfg = replace(cfg, channel_timeout=0.05)
    samples = by_name(await collect(guild, opts, fast_cfg, analyzer))
    assert samples['slowpoke'].status == 'skipped'
    assert samples['slowpoke'].reason == 'timeout'
    assert samples['general'].status == 'ok'


async def test_global_deadline_marks_rest_skipped(guild, opts, cfg, analyzer):
    for i in range(5):
        guild.text_channels.append(
            make_channel(f'slow-{i}', 200 + i,
                         [(guild._members[101], 'x', 5)],
                         per_message_sleep=0.2))
    tight = replace(cfg, max_seconds=0.15, channel_timeout=5.0, concurrency=1)
    samples = await collect(guild, opts, tight, analyzer)
    statuses = {s.status for s in samples}
    reasons = {s.reason for s in samples if s.status == 'skipped'}
    assert 'skipped' in statuses
    assert 'deadline' in reasons or 'timeout' in reasons
    # every channel still produced a sample — nothing vanished
    assert len(samples) == len(guild.text_channels)


async def test_channel_subset_selection(guild, opts, cfg, analyzer):
    subset = replace(opts, channels=('general',))
    samples = await collect(guild, subset, cfg, analyzer)
    assert [s.name for s in samples] == ['general']


async def test_window_filtering(guild, opts, cfg, analyzer, members):
    from datetime import datetime, timedelta, timezone
    old = make_channel('oldies', 95, [(
        members['alice'], 'ancient history',
        int((timedelta(hours=48).total_seconds()) // 60))])
    guild.text_channels.append(old)
    samples = by_name(await collect(guild, opts, cfg, analyzer))  # window: 6h
    assert samples['oldies'].msg_count == 0           # outside `after=` window


async def test_no_orphan_tasks(guild, opts, cfg, analyzer):
    before = len(asyncio.all_tasks())
    await collect(guild, opts, cfg, analyzer)
    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) <= before + 1
