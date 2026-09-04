"""Fakes for offline snapshot tests — no Discord gateway, no Ollama server."""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import discord
import pytest

os.environ.setdefault('DB_PATH', tempfile.mktemp(suffix='.db'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agent import PsychologicalAnalyzer  # noqa: E402


# ---------------- fakes ----------------

class FakeMember:
    def __init__(self, id, name, bot=False, joined_days_ago=365, manage_messages=False):
        self.id = id
        self.display_name = name
        self.bot = bot
        self.joined_at = datetime.now(timezone.utc) - timedelta(days=joined_days_ago)
        self.guild_permissions = SimpleNamespace(manage_messages=manage_messages)


class FakeMessage:
    _next_id = 1000

    def __init__(self, author, content, created_at=None):
        FakeMessage._next_id += 1
        self.id = FakeMessage._next_id
        self.author = author
        self.content = content
        self.created_at = created_at or datetime.now(timezone.utc)


class FakeResponse:
    def __init__(self, status=403, reason='Forbidden'):
        self.status = status
        self.reason = reason


class FakeChannel:
    def __init__(self, name, id, messages=None, *, category=None,
                 view=True, history_perm=True, raise_exc=None, raise_once=False,
                 per_message_sleep=0.0):
        self.id = id
        self.name = name
        self.category = SimpleNamespace(name=category) if category else None
        self.topic = None
        self._messages = messages or []
        self._view = view
        self._history_perm = history_perm
        self._raise_exc = raise_exc
        self._raise_once = raise_once
        self._sleep = per_message_sleep
        self.history_calls = 0

    def permissions_for(self, _me):
        return SimpleNamespace(view_channel=self._view,
                               read_message_history=self._history_perm)

    def history(self, *, limit=None, after=None, oldest_first=None):
        self.history_calls += 1
        if self._raise_exc is not None and not (self._raise_once and self.history_calls > 1):
            exc = self._raise_exc

            async def boom():
                raise exc
                yield  # pragma: no cover
            return boom()

        async def gen():
            msgs = [m for m in self._messages
                    if after is None or m.created_at > after]
            msgs.sort(key=lambda m: m.created_at, reverse=not bool(oldest_first))
            if limit is not None:
                msgs = msgs[:limit]
            for m in msgs:
                if self._sleep:
                    await asyncio.sleep(self._sleep)
                yield m
        return gen()


class FakeGuild:
    def __init__(self, name='No Sleep Zone', id=12345, channels=None, members=None):
        self.id = id
        self.name = name
        self.text_channels = channels or []
        self.me = FakeMember(id=999999, name='Acheron', bot=True)
        self._members = {m.id: m for m in (members or [])}

    def get_member(self, id):
        return self._members.get(id)


class FakeConnector:
    """Stands in for OllamaConnector. Never does network I/O."""

    FALLBACKS = ("my brain just blue-screened, say that again 💀",)

    def __init__(self, response='server looks alive', *, model='fake-model',
                 fail_with_fallback=False, record=True):
        self.response = response
        self.model = model
        self.fail_with_fallback = fail_with_fallback
        self.calls = [] if record else None

    async def generate(self, messages, num_predict=None, temperature=None,
                       think=None, timeout=None, model=None):
        if self.calls is not None:
            self.calls.append({
                'messages': messages, 'num_predict': num_predict,
                'temperature': temperature, 'think': think,
                'timeout': timeout, 'model': model,
            })
        if self.fail_with_fallback:
            return self.FALLBACKS[0]
        return self.response


# ---------------- shared fixtures ----------------

@pytest.fixture
def analyzer():
    return PsychologicalAnalyzer()


@pytest.fixture
def cfg():
    from snapshot.options import SnapshotConfig
    return SnapshotConfig()


@pytest.fixture
def opts(cfg):
    from snapshot.options import SnapshotOptions
    return SnapshotOptions(hours=6, detail='medium', channels=(), top=12,
                           fresh=True, include_bots=False)


def make_channel(name, id, pairs, **kwargs):
    """pairs: list of (FakeMember, content, minutes_ago)"""
    now = datetime.now(timezone.utc)
    messages = [FakeMessage(author, content, now - timedelta(minutes=mins))
                for author, content, mins in pairs]
    return FakeChannel(name, id, messages, **kwargs)


@pytest.fixture
def members():
    return {
        'alice': FakeMember(101, 'alice'),
        'bob': FakeMember(102, 'bob'),
        'carol': FakeMember(103, 'carol', joined_days_ago=2),  # new voice
        'botfriend': FakeMember(104, 'botfriend', bot=True),
    }


@pytest.fixture
def guild(members):
    alice, bob, carol, botfriend = (members['alice'], members['bob'],
                                    members['carol'], members['botfriend'])
    channels = [
        make_channel('general', 1, [
            (alice, 'this server is amazing, love it', 10),
            (bob, 'yeah best community', 20),
            (alice, 'game night was incredible', 30),
            (carol, 'glad I joined, you all rock', 40),
            (alice, 'tonight was so fun', 50),
            (botfriend, 'automated bot noise', 15),
            (alice, '', 25),  # empty content: counts as activity, no sentiment
        ]),
        make_channel('venting', 2, [
            (bob, 'awful terrible horrible day', 60),
            (bob, 'everything is the worst, i hate this', 70),
            (carol, 'bad week honestly, really sad', 80),
            (bob, ' miserable. garbage. ugh', 90),
            (carol, 'cryyying this sucks so bad', 100),
            (bob, 'worst week of my life fr', 110),
        ]),
        make_channel('quiet-corner', 3, []),
        FakeChannel('locked-ops', 4, [], view=False),  # no_access, no API call
    ]
    return FakeGuild(channels=channels, members=list(members.values()))
