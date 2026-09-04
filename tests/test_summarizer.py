"""Think-tag stripping, banned-token screen, failure → None, disabled → None."""

from dataclasses import replace
from datetime import datetime, timezone

from snapshot.metrics import ChannelMetrics, ServerSnapshot
from snapshot.summarizer import build_prompt, postprocess, summarize
from tests.conftest import FakeConnector


def make_snapshot():
    return ServerSnapshot(
        guild_name='No Sleep Zone', window_hours=6,
        taken_at=datetime.now(timezone.utc),
        channels=[
            ChannelMetrics('general', 47, 12, 0.1, 'chill/neutral', 'hot', False,
                           [(101, 30), (102, 17)]),
            ChannelMetrics('memes', 31, 9, 0.6, 'electric', 'warm', True, [(101, 31)]),
            ChannelMetrics('quiet-corner', 0, 0, None, 'n/a', 'quiet', False, []),
        ],
        scanned=3, skipped=0, skipped_reasons={}, total_msgs=78,
        total_unique_authors=14,
        highlights=['#general — 47 msgs, 12 voices', '📈 #memes heating up in the last hour'],
        partial=False, elapsed_s=2.5, overall_mood='good vibes',
    )


def test_prompt_is_numbers_only_and_compact(cfg, opts):
    msgs = build_prompt(make_snapshot(), opts)
    assert msgs[0]['role'] == 'system'
    user = msgs[1]['content']
    assert '#general | 47 | 12 |' in user
    assert 'burst' in user
    assert 'HIGHLIGHTS:' in user
    assert len(user) < 1800 + 400      # table cap + header lines
    # persona contract present in system prompt
    assert 'pet names' in msgs[0]['content']
    assert "can't see" in msgs[0]['content']


def test_postprocess_strips_think_and_fences():
    raw = "<think>let me think about this table deeply\nyes hmm</think>\n```\n" \
          "server's alive\n#general is moving\n```"
    out = postprocess(raw)
    assert '<think>' not in out and '```' not in out
    assert "server's alive" in out


def test_postprocess_strips_role_labels():
    out = postprocess("Acheron: server looks busy tonight and alive")
    assert out == "server looks busy tonight and alive"


def test_banned_tokens_drop_lines():
    raw = "the server is vibing\nokay babe here's the deal\n#memes is popping\n" \
          "I can't see the channels\n#general steady"
    out = postprocess(raw)
    assert 'babe' not in out
    assert "can't see" not in out.lower()
    assert '#memes is popping' in out
    assert '#general steady' in out


def test_all_banned_or_short_returns_none():
    assert postprocess("hey babe") is None          # only banned content
    assert postprocess("ok") is None                # < 20 chars
    assert postprocess("") is None


def test_six_line_cap():
    raw = "\n".join(f"line number {i} with enough chars" for i in range(10))
    assert len(postprocess(raw).splitlines()) <= 6


async def test_summarize_happy_path(cfg, opts):
    conn = FakeConnector("#general is the spot tonight\nquiet otherwise")
    out = await summarize(make_snapshot(), opts, cfg, conn)
    assert out and '#general' in out
    call = conn.calls[0]
    assert call['num_predict'] == cfg.llm_num_predict
    assert call['temperature'] == 0.3
    assert call['think'] is False
    assert call['timeout'] == cfg.llm_timeout


async def test_summarize_disabled_returns_none(cfg, opts):
    cfg_off = replace(cfg, llm_enabled=False)
    conn = FakeConnector("should never be used")
    assert await summarize(make_snapshot(), opts, cfg_off, conn) is None
    assert conn.calls == []


async def test_connector_failure_text_becomes_none(cfg, opts):
    conn = FakeConnector(fail_with_fallback=True)
    assert await summarize(make_snapshot(), opts, cfg, conn) is None


async def test_garbage_output_becomes_none(cfg, opts):
    conn = FakeConnector("<think>only thoughts, no answer</think>")
    assert await summarize(make_snapshot(), opts, cfg, conn) is None


async def test_snapshot_model_override_passed_through(cfg, opts):
    cfg_model = replace(cfg, llm_model='qwen3.5:latest')
    conn = FakeConnector("fine summary here for you")
    await summarize(make_snapshot(), opts, cfg_model, conn)
    # connector.generate receives it; OllamaConnector maps it onto client.chat
    assert conn.calls[0]['num_predict'] == 220
