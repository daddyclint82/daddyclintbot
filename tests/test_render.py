"""Embed limits, trim order, plain-text fallback, non-owner never sees names."""

from datetime import datetime, timezone

from snapshot.metrics import ChannelMetrics, ServerSnapshot
from snapshot.render import build


def make_snapshot(n_channels=5, long_names=False, highlights=True):
    channels = []
    for i in range(n_channels):
        name = (f'a-very-long-channel-name-number-{i:02d}' if long_names
                else f'channel-{i:02d}')
        tier = 'hot' if i == 0 else ('warm' if i < 3 else ('low' if i < 5 else 'quiet'))
        count = max(40 - i * 5, 0)
        channels.append(ChannelMetrics(
            name=name, msg_count=count, unique_authors=max(10 - i, 0),
            avg_compound=0.1 if count else None,
            mood='chill/neutral' if count else 'n/a',
            activity_tier=tier, burst=(i == 1),
            top_authors=[(101, count)] if count else [],
        ))
    return ServerSnapshot(
        guild_name='No Sleep Zone', window_hours=6,
        taken_at=datetime.now(timezone.utc),
        channels=channels, scanned=len(channels), skipped=0, skipped_reasons={},
        total_msgs=sum(c.msg_count for c in channels), total_unique_authors=10,
        highlights=(['#channel-00 — 40 msgs, 10 voices',
                     '📈 #channel-01 heating up in the last hour',
                     '🔥 #channel-00 is hyped'] if highlights else []),
        partial=False, elapsed_s=3.2, overall_mood='good vibes',
    )


def embed_size(embed):
    size = len(embed.title or '') + len(embed.description or '')
    if embed.footer and embed.footer.text:
        size += len(embed.footer.text)
    for f in embed.fields:
        size += len(f.name) + len(f.value)
    return size


def field(embed, name):
    return next((f for f in embed.fields if f.name == name), None)


def test_embed_structure_owner_medium(cfg, opts, guild):
    out = build(make_snapshot(), 'the place is alive tonight',
                opts=opts, cfg=cfg, guild=guild, is_owner=True,
                model_name='qwen3.5:latest')
    embed = out.embed
    assert embed.title.startswith('📸 NSZ Snapshot')
    assert 'Window: last 6h' in embed.description
    assert field(embed, 'Channels') is not None
    assert field(embed, 'Highlights') is not None
    assert field(embed, 'Read').value == 'the place is alive tonight'
    assert 'qwen3.5:latest' in embed.footer.text
    assert 'code block' in field(embed, 'Channels').value or '```' in field(embed, 'Channels').value
    assert out.plain_text


def test_non_owner_is_count_only_no_names(cfg, opts, guild):
    from snapshot.options import SnapshotOptions
    high = SnapshotOptions(hours=6, detail='high', channels=(), top=12,
                           fresh=True, include_bots=False)
    out = build(make_snapshot(), 'alive', opts=high, cfg=cfg, guild=guild,
                is_owner=False)
    embed = out.embed
    assert field(embed, 'Top voices') is None          # names never render
    assert field(embed, 'Highlights') is None          # forced count-only
    channels_val = field(embed, 'Channels').value
    assert 'voices' in channels_val
    assert 'chill' not in channels_val                 # no mood column at low detail
    assert 'alice' not in str(embed.to_dict())


def test_owner_high_gets_top_voices(cfg, guild, members):
    from snapshot.options import SnapshotOptions
    high = SnapshotOptions(hours=6, detail='high', channels=(), top=12,
                           fresh=True, include_bots=False)
    out = build(make_snapshot(), 'x' * 30, opts=high, cfg=cfg, guild=guild,
                is_owner=True)
    voices = field(out.embed, 'Top voices')
    assert voices is not None
    assert '@alice' in voices.value                    # resolved from guild cache
    assert '×' in voices.value


def test_uncached_member_falls_back_to_id(cfg, guild):
    from snapshot.options import SnapshotOptions
    high = SnapshotOptions(hours=6, detail='high', channels=(), top=12,
                           fresh=True, include_bots=False)
    snap = make_snapshot()
    snap.channels[0].top_authors = [(424242, 40)]      # not in guild cache
    out = build(snap, 'x' * 30, opts=high, cfg=cfg, guild=guild, is_owner=True)
    assert 'user-424242' in field(out.embed, 'Top voices').value


def test_fallback_read_when_no_summary(cfg, opts, guild):
    out = build(make_snapshot(), None, opts=opts, cfg=cfg, guild=guild,
                is_owner=True)
    assert '(AI read unavailable)' in field(out.embed, 'Read').value
    assert 'mood' in field(out.embed, 'Read').value


def test_partial_banner(cfg, opts, guild):
    snap = make_snapshot()
    snap.partial = True
    snap.skipped = 2
    snap.skipped_reasons = {'no_access': 2}
    out = build(snap, 'x' * 30, opts=opts, cfg=cfg, guild=guild, is_owner=True)
    assert '⚠️ Partial' in out.embed.description


def test_cache_footer_and_fresh_hint(cfg, opts, guild):
    out = build(make_snapshot(), 'x' * 30, opts=opts, cfg=cfg, guild=guild,
                is_owner=True, cache_age=45)
    assert 'cache 45s' in out.embed.footer.text
    assert 'fresh' in out.embed.footer.text


def test_ignored_tokens_in_footer(cfg, opts, guild):
    out = build(make_snapshot(), 'x' * 30, opts=opts, cfg=cfg, guild=guild,
                is_owner=True, ignored=['bogus', 'wat:7'])
    assert 'ignored: bogus, wat:7' in out.embed.footer.text


def test_discord_limits_enforced(cfg, guild):
    from snapshot.options import SnapshotOptions
    big = SnapshotOptions(hours=6, detail='high', channels=(), top=25,
                          fresh=True, include_bots=False)
    snap = make_snapshot(n_channels=25, long_names=True)
    out = build(snap, 'word ' * 150, opts=big, cfg=cfg, guild=guild, is_owner=True)
    embed = out.embed
    assert embed_size(embed) <= 6000
    assert len(embed.fields) <= 25
    for f in embed.fields:
        assert len(f.value) <= 1024
    assert len(embed.description or '') <= 4096


def test_trim_order_top_voices_first(cfg, guild):
    from snapshot.options import SnapshotOptions
    big = SnapshotOptions(hours=6, detail='high', channels=(), top=25,
                          fresh=True, include_bots=False)
    snap = make_snapshot(n_channels=25, long_names=True)
    out = build(snap, 'word ' * 150, opts=big, cfg=cfg, guild=guild, is_owner=True)
    # Top voices must be sacrificed before Channels content
    if len(out.embed.fields) < 4:
        assert field(out.embed, 'Top voices') is None
    assert field(out.embed, 'Channels') is not None


def test_plain_text_fallback_always_built(cfg, opts, guild):
    out = build(make_snapshot(), None, opts=opts, cfg=cfg, guild=guild,
                is_owner=True, cache_age=12, ignored=['nah'])
    text = out.plain_text
    assert '📸 NSZ Snapshot' in text
    assert 'Window: last 6h' in text
    assert 'cached 12s' in text
    assert 'ignored: nah' in text
