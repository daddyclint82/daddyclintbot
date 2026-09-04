"""Tiers, mood gating, burst, highlights ordering, zero-channel case. Pure."""

from snapshot.metrics import ChannelMetrics, ChannelSample, compute


def make_sample(name, id, n_msgs, compounds=None, authors=None,
                last_sixth=0, status='ok', reason=None):
    compounds = compounds if compounds is not None else [0.0] * n_msgs
    authors = authors or {1000 + i: 1 for i in range(min(n_msgs, 3))}
    return ChannelSample(
        channel_id=id, name=name, category=None, status=status, reason=reason,
        msg_count=n_msgs, unique_authors=len(authors), compounds=compounds,
        author_counts=dict(authors), last_sixth_count=last_sixth,
    )


def test_tier_thresholds_scale_with_window(cfg, opts):
    samples = [
        make_sample('hot-ch', 1, 35),      # ≥ 30 @ 6h
        make_sample('warm-ch', 2, 12),     # ≥ 10 @ 6h
        make_sample('low-ch', 3, 2),
        make_sample('quiet-ch', 4, 0),
    ]
    snap = compute(samples, opts, cfg, guild=None, guild_name='NSZ', elapsed_s=1.0)
    tiers = {c.name: c.activity_tier for c in snap.channels}
    assert tiers == {'hot-ch': 'hot', 'warm-ch': 'warm',
                     'low-ch': 'low', 'quiet-ch': 'quiet'}


def test_tier_scales_for_bigger_window(cfg):
    from snapshot.options import SnapshotOptions
    opts24 = SnapshotOptions(hours=24, detail='medium', channels=(), top=12,
                             fresh=True, include_bots=False)
    # 30 msgs over 24h is NOT hot (needs 30 * 24/6 = 120)
    samples = [make_sample('busy', 1, 40), make_sample('mega', 2, 130)]
    snap = compute(samples, opts24, cfg, guild=None, guild_name='NSZ', elapsed_s=1.0)
    tiers = {c.name: c.activity_tier for c in snap.channels}
    assert tiers['busy'] == 'warm'
    assert tiers['mega'] == 'hot'


def test_mood_gating_below_min_samples(cfg, opts):
    # default min_msgs_for_mood = 3 → 2 compounds means mood n/a
    samples = [make_sample('thin', 1, 2, compounds=[0.9, 0.9])]
    snap = compute(samples, opts, cfg, guild=None, guild_name='NSZ', elapsed_s=0.5)
    assert snap.channels[0].mood == 'n/a'
    assert snap.channels[0].avg_compound is None


def test_burst_detection(cfg, opts):
    # 6 msgs, 3 in the last sixth (≥ 0.4 * 6) → burst
    samples = [make_sample('popping', 1, 6, last_sixth=3)]
    snap = compute(samples, opts, cfg, guild=None, guild_name='NSZ', elapsed_s=0.5)
    assert snap.channels[0].burst is True
    assert any('heating up' in h for h in snap.highlights)


def test_no_burst_under_msg_floor(cfg, opts):
    samples = [make_sample('small', 1, 4, last_sixth=4)]
    snap = compute(samples, opts, cfg, guild=None, guild_name='NSZ', elapsed_s=0.5)
    assert snap.channels[0].burst is False


def test_highlights_order_and_content(cfg, opts, members):
    from tests.conftest import FakeGuild
    samples = [
        make_sample('general', 1, 40, compounds=[0.6] * 40, authors={101: 40}),
        make_sample('vents', 2, 8, compounds=[-0.5] * 8, authors={102: 8}),
        make_sample('dead', 3, 0),
        make_sample('newbie-chat', 4, 5, compounds=[0.1] * 5,
                    authors={103: 5}),  # carol joined 2 days ago
    ]
    guild = FakeGuild(members=list(members.values()))
    snap = compute(samples, opts, cfg, guild=guild, guild_name='NSZ',
                   elapsed_s=1.0, for_owner=True)
    assert snap.highlights[0].startswith('#general')           # hottest first
    assert any('#general is hyped' in h for h in snap.highlights)
    assert any('#vents is tense' in h for h in snap.highlights)
    assert any('quiet' in h for h in snap.highlights)
    assert any('new voices' in h and 'carol' in h for h in snap.highlights)
    assert len(snap.highlights) <= 6
    assert all(len(h) <= 90 for h in snap.highlights)


def test_new_voices_count_only_for_non_owner(cfg, opts, members):
    from tests.conftest import FakeGuild
    samples = [make_sample('ch', 1, 5, authors={103: 5})]
    guild = FakeGuild(members=list(members.values()))
    snap = compute(samples, opts, cfg, guild=guild, guild_name='NSZ',
                   elapsed_s=1.0, for_owner=False)
    new_voice = [h for h in snap.highlights if 'new voices' in h]
    assert new_voice and 'carol' not in new_voice[0] and '1' in new_voice[0]


def test_zero_channels_case(cfg, opts):
    snap = compute([], opts, cfg, guild=None, guild_name='NSZ', elapsed_s=0.1)
    assert snap.total_msgs == 0
    assert snap.total_unique_authors == 0
    assert snap.highlights == []
    assert snap.overall_mood == 'n/a'


def test_skipped_channels_counted_and_last(cfg, opts):
    samples = [
        make_sample('ok-ch', 1, 5),
        make_sample('locked', 2, 0, status='skipped', reason='no_access'),
    ]
    snap = compute(samples, opts, cfg, guild=None, guild_name='NSZ', elapsed_s=1.0)
    assert snap.partial is True
    assert snap.skipped == 1
    assert snap.skipped_reasons == {'no_access': 1}
    assert snap.channels[-1].activity_tier == 'skipped'
    assert any('skipped' in h for h in snap.highlights)


def test_totals_and_sorting(cfg, opts):
    samples = [make_sample('b', 1, 5, authors={1: 3, 2: 2}),
               make_sample('a', 2, 10, authors={2: 10})]  # author 2 overlaps
    snap = compute(samples, opts, cfg, guild=None, guild_name='NSZ', elapsed_s=2.0)
    assert snap.total_msgs == 15
    assert snap.total_unique_authors == 2        # union, not sum
    assert snap.channels[0].name == 'a'          # busiest first
    assert snap.elapsed_s == 2.0
