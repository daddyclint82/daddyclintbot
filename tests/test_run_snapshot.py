"""Orchestration: access control, cache, breaker, in-flight, never-silent."""

import asyncio
from types import SimpleNamespace

import pytest

import snapshot as pkg
from snapshot import run_snapshot
from tests.conftest import FakeConnector, FakeMember
from agent import PsychologicalAnalyzer


@pytest.fixture(autouse=True)
def reset_snapshot_state(monkeypatch, tmp_path):
    """Every test gets a fresh config/cache/breaker/in-flight table."""
    pkg._config = None
    pkg._cache = None
    pkg._breaker = None
    pkg._inflight.clear()
    monkeypatch.setenv('SNAPSHOT_LLM_ENABLED', 'true')
    yield
    pkg._inflight.clear()


@pytest.fixture
def engine():
    return SimpleNamespace(analyzer=PsychologicalAnalyzer(), llm=FakeConnector(
        "#general is carrying the place\nquiet night otherwise"))


@pytest.fixture
def owner():
    return FakeMember(777, 'clint', manage_messages=True)


@pytest.fixture
def regular():
    return FakeMember(888, 'dave', manage_messages=False)


async def test_owner_happy_path(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    out = await run_snapshot(guild=guild, author=owner, is_owner=True,
                             raw_args='', engine=engine)
    assert out.embed is not None
    assert field_value(out.embed, 'Channels')
    assert field_value(out.embed, 'Read')
    assert engine.llm.calls                          # LLM was used


async def test_non_owner_refused_when_owner_only(guild, regular, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    out = await run_snapshot(guild=guild, author=regular, is_owner=False,
                             raw_args='', engine=engine)
    assert out.embed is None
    assert out.plain_text                            # in-voice refusal, not silence
    assert all(c.history_calls == 0 for c in guild.text_channels)  # no API hit


async def test_everyone_access_gives_count_only(guild, regular, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'everyone')
    out = await run_snapshot(guild=guild, author=regular, is_owner=False,
                             raw_args='detail:high', engine=engine)
    assert out.embed is not None
    assert field_value(out.embed, 'Top voices') is None
    assert 'alice' not in str(out.embed.to_dict())


async def test_staff_access_allows_manage_messages(guild, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'staff')
    mod = FakeMember(889, 'modmorgan', manage_messages=True)
    out = await run_snapshot(guild=guild, author=mod, is_owner=False,
                             raw_args='', engine=engine)
    assert out.embed is not None


async def test_cache_hit_then_fresh_bypass(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    first = await run_snapshot(guild=guild, author=owner, is_owner=True,
                               raw_args='', engine=engine)
    calls_after_first = sum(c.history_calls for c in guild.text_channels)
    assert calls_after_first > 0

    second = await run_snapshot(guild=guild, author=owner, is_owner=True,
                                raw_args='', engine=engine)
    assert 'cache' in (second.embed.footer.text or '')
    assert sum(c.history_calls for c in guild.text_channels) == calls_after_first

    third = await run_snapshot(guild=guild, author=owner, is_owner=True,
                               raw_args='fresh', engine=engine)
    assert 'cache' not in (third.embed.footer.text or '')
    assert sum(c.history_calls for c in guild.text_channels) > calls_after_first


async def test_llm_disabled_still_delivers(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    monkeypatch.setenv('SNAPSHOT_LLM_ENABLED', 'false')
    out = await run_snapshot(guild=guild, author=owner, is_owner=True,
                             raw_args='', engine=engine)
    assert out.embed is not None
    assert '(AI read unavailable)' in field_value(out.embed, 'Read')


async def test_llm_failure_still_delivers(guild, owner, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    engine = SimpleNamespace(analyzer=PsychologicalAnalyzer(),
                             llm=FakeConnector(fail_with_fallback=True))
    out = await run_snapshot(guild=guild, author=owner, is_owner=True,
                             raw_args='', engine=engine)
    assert out.embed is not None
    assert '(AI read unavailable)' in field_value(out.embed, 'Read')


async def test_never_silent_on_internal_explosion(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')

    async def boom(*a, **k):
        raise RuntimeError('the floor is lava')

    monkeypatch.setattr(pkg.collector, 'collect', boom)
    out = await run_snapshot(guild=guild, author=owner, is_owner=True,
                             raw_args='', engine=engine)
    assert out.plain_text                              # apology, not silence
    assert out.embed is None


async def test_breaker_trips_after_three_failures(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')

    async def boom(*a, **k):
        raise RuntimeError('nope')

    monkeypatch.setattr(pkg.collector, 'collect', boom)
    for _ in range(3):
        await run_snapshot(guild=guild, author=owner, is_owner=True,
                           raw_args='fresh', engine=engine)
    out = await run_snapshot(guild=guild, author=owner, is_owner=True,
                             raw_args='fresh', engine=engine)
    assert 'cooling down' in out.plain_text


async def test_zero_readable_channels(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    for ch in guild.text_channels:
        ch._history_perm = False                        # revoke everywhere
    out = await run_snapshot(guild=guild, author=owner, is_owner=True,
                             raw_args='fresh', engine=engine)
    assert out.embed is None
    assert '0 channels readable' in out.plain_text


async def test_inflight_second_caller_told_cooking(guild, owner, engine, monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'owner')
    real_collect = pkg.collector.collect

    async def slow_collect(*a, **k):
        await asyncio.sleep(0.3)
        return await real_collect(*a, **k)

    monkeypatch.setattr(pkg.collector, 'collect', slow_collect)
    first = asyncio.create_task(run_snapshot(
        guild=guild, author=owner, is_owner=True, raw_args='hours:6', engine=engine))
    await asyncio.sleep(0.05)
    second = await run_snapshot(guild=guild, author=owner, is_owner=True,
                                raw_args='hours:24', engine=engine)   # different args
    assert 'cooking' in second.plain_text
    assert (await first).embed is not None


def field_value(embed, name):
    f = next((f for f in embed.fields if f.name == name), None)
    return f.value if f else None
