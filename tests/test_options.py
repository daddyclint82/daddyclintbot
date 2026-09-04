"""Grammar, clamping, ignored tokens, env config validation. Pure."""

from snapshot.options import SnapshotConfig, parse_args


def test_defaults(cfg):
    opts, ignored = parse_args('', cfg)
    assert opts.hours == 6
    assert opts.detail == 'medium'
    assert opts.top == 12
    assert opts.fresh is False
    assert opts.channels == ()
    assert ignored == []


def test_full_grammar(cfg):
    opts, ignored = parse_args('hours:24 channels:general,memes detail:high top:8 fresh', cfg)
    assert opts.hours == 24
    assert opts.detail == 'high'
    assert opts.top == 8
    assert opts.fresh is True
    assert opts.channels == ('general', 'memes')
    assert ignored == []


def test_channels_strip_hash_and_spaces(cfg):
    opts, _ = parse_args('channels:#general, memes ,#vibe-check top:9', cfg)
    assert opts.channels == ('general', 'memes', 'vibe-check')
    assert opts.top == 9  # parsing resumes after the spacey channel list


def test_clamps_not_rejects(cfg):
    opts, _ = parse_args('hours:0', cfg)
    assert opts.hours == 1
    opts, _ = parse_args('hours:9999', cfg)
    assert opts.hours == 168
    opts, _ = parse_args('top:99', cfg)
    assert opts.top == 25
    opts, _ = parse_args('top:1', cfg)
    assert opts.top == 3


def test_invalid_values_fall_back(cfg):
    opts, _ = parse_args('hours:banana', cfg)
    assert opts.hours == 6  # default, not a crash


def test_unknown_tokens_ignored_and_listed(cfg):
    opts, ignored = parse_args('hours:12 wat:7 bogus detail:ultra', cfg)
    assert opts.hours == 12
    assert opts.detail == 'medium'  # 'ultra' invalid → default, token ignored
    assert 'wat:7' in ignored and 'bogus' in ignored and 'detail:ultra' in ignored


def test_env_config_clamps_and_never_raises(monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'overlord')       # invalid
    monkeypatch.setenv('SNAPSHOT_DEFAULT_HOURS', '0')       # below range
    monkeypatch.setenv('SNAPSHOT_CONCURRENCY', 'banana')    # not a number
    monkeypatch.setenv('SNAPSHOT_LLM_TIMEOUT', '99999')     # above range
    cfg = SnapshotConfig.from_env()
    assert cfg.access == 'owner'
    assert cfg.default_hours == 1
    assert cfg.concurrency == 4
    assert cfg.llm_timeout == 300.0


def test_env_config_reads_good_values(monkeypatch):
    monkeypatch.setenv('SNAPSHOT_ACCESS', 'staff')
    monkeypatch.setenv('SNAPSHOT_DEFAULT_DETAIL', 'high')
    monkeypatch.setenv('SNAPSHOT_INCLUDE_BOTS', 'true')
    monkeypatch.setenv('SNAPSHOT_MODEL', 'qwen3.5:latest')
    cfg = SnapshotConfig.from_env()
    assert cfg.access == 'staff'
    assert cfg.default_detail == 'high'
    assert cfg.include_bots is True
    assert cfg.llm_model == 'qwen3.5:latest'
