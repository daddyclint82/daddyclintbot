# Engineering Brief — `!snapshot`: On-Demand Server Awareness for Acheron

**Repo:** `daddyclint82/daddyclintbot` · **Target branch:** `main` · **Brief version:** 1.0 (2026-09-04)
**Audience:** an implementing LLM/engineer with repo access. This document is the complete spec. Read it fully before writing code.

> **How to use this file:** paste the whole thing (or point the model at this path) and say:
> *"Implement Section 4–10 exactly as specified. Do not modify anything listed in Section 3.2. Return a single reviewable diff plus the test run output."*
> A ready-to-paste kickoff prompt is in Appendix A.

---

## 0. TL;DR

Add **one** new command, `!snapshot`, that — **only when asked** — walks every text channel the bot can see, computes per-channel activity + sentiment aggregates in-process, has the LLM write a short "what's going on in the No Sleep Zone" summary, and returns one Discord embed. Then it forgets everything except a 5-minute in-memory cache.

It must be **additive** (no existing behavior changes), **modular** (new `src/snapshot/` package, ~5 small files; `discord_bot.py` grows by one command method + one tuple entry), **self-healing** (every failure degrades to a partial-but-useful reply; the bot never crashes, never sends nothing), and **local-model-friendly** (runs well on `qwen3.5:latest` through Ollama; the LLM is the *last* step and is fully optional at runtime).

---

## 1. Mission & Non-Goals

### 1.1 Mission
Give the server owner (and optionally staff) a **point-in-time snapshot** of the whole Discord server, on demand:
- which channels are active, how much, how many distinct people
- the sentiment per channel (aggregate, not per person)
- rule-based highlights (bursts, sentiment extremes, new voices, dead zones)
- a 3–6 line human summary in the bot's existing voice

### 1.2 Explicit non-goals (do NOT build these)
| Not building | Why |
|---|---|
| Continuous / live monitoring, event listeners for analytics, new `tasks.loop` | Owner explicitly rejected always-on observation. Snapshot is pull-only. |
| Per-user behavioral profiles, footprints, presence tracking, "sleep heatmaps" | Privacy floor. Aggregates only. |
| Anomaly alerts / DMs to owner on sentiment shift | That is live monitoring by another name. |
| New SQLite tables for snapshot data | v1 is in-memory only. (v2 idea: opt-in aggregate persistence for `!snapshot compare`. Out of scope.) |
| Reading threads, forum posts, voice-text, or DMs | v1 scope = `guild.text_channels` (includes announcement/news channels). Count what you skip; do not read it. |
| Selfbot / user-token techniques (as in Privex-chat/sentinel's shipped data layer) | ToS violation. This bot uses a bot token, gateway intents, and role permissions only. |
| A web dashboard, REST API, SSE stream, NDJSON export | Discord embed is the only output surface in v1. |

**What we borrow from the Sentinel architecture:** local-first, no telemetry, "collect → compute → summarize" pipeline, and a transient cache so repeated asks don't re-hit the API. Nothing else.

---

## 2. Repo Facts (verified live 2026-09-04 — do not assume otherwise)

### 2.1 Layout
```
daddyclintbot/
├── .env                      # secrets + config (never commit; mode 600)
├── .env.example              # documented keys, no values  ← you will extend this
├── .gitignore
├── Modelfile                 # FROM qwen3.5:4b, num_ctx 16384 (reference only)
├── README.md
├── SETUP_NOTES.md
├── daddyclintbot.service     # systemd unit (documented, not yet deployed)
├── requirements.txt
├── config/server_knowledge.md
├── docs/SNAPSHOT_ENGINEERING_BRIEF.md   ← this file
└── src/
    ├── agent.py              # ~1,050 lines: engine, DB, analyzer, Ollama connector, prompts
    └── discord_bot.py        # ~800 lines: Bot subclass, handlers, commands, loops
```

### 2.2 Stack
- Python 3.12 · `discord.py` **2.7.1** · `ollama` **0.6.2** (python client) · `vaderSentiment` · `python-dotenv`
- LLM host: Ollama over HTTP (`OLLAMA_HOST`), currently `minimax-m3:cloud`; **target for this feature: `qwen3.5:latest`** (6.6 GB, local, confirmed present on the host)
- Storage: SQLite via `DatabaseManager` (`DB_PATH`)
- Process model: single foreground process; **only one gateway connection per bot token** (a second instance with the same token stalls — stop the running bot before test-running another)

### 2.3 Existing interfaces you will reuse (real signatures)

`src/agent.py`
```python
class PsychologicalAnalyzer:
    def analyze(self, text: str) -> Dict
    # returns {'compound_score': float, 'acr_trigger': bool, 'vulnerability_score': float,
    #          'positivity': float, 'negativity': float, 'neutrality': float}

class OllamaConnector:
    # attrs: model, host, temperature, timeout, num_predict, num_predict_owner, max_retries,
    #        last_success_at, last_latency, total_failures, total_generations
    def check_connection(self) -> bool
    async def generate(self, messages: List[Dict[str, str]], num_predict: int = None) -> str
    # "Retries with backoff, never raises." Returns a fallback string on total failure.

class DaddyClintBot:                       # the engine; exposed as bot.engine
    self.db: DatabaseManager
    self.analyzer: PsychologicalAnalyzer
    self.ollama: OllamaConnector
    @staticmethod
    def _mood_label(compound: float) -> str      # reuse for consistent sentiment labels
    async def process_message(self, user_id, user_name, message, is_owner=False,
                              force_intent=None, extra_directive=None,
                              num_predict_override=None) -> Tuple[str, Dict]
```

`src/discord_bot.py` — `class DaddyClintDiscordBot(commands.Bot)`
```python
def _is_owner(self, author) -> bool                 # str(author.id) == OWNER_ID
async def _keep_typing(self, channel)               # typing indicator helper
@staticmethod
async def _human_typing_delay(response: str)        # max(0.6s, min(2.5s, len*0.035))
@staticmethod
def _chunk(text: str, size: int = 1990)             # split for 2000-char limit
def _build_channel_directory(self)
async def _gather_server_intel(self)                # reads rules/info channels for the persona (leave alone)
```

### 2.4 Command registration pattern (mandatory — discord.py 2.7.1 quirk)
Commands are declared as `@commands.command(...)` on the class body **and** explicitly registered in `setup_hook()`; the decorator does not auto-register on a `Bot` subclass and the raw callback has a `(self, ctx)` signature that must be bound:

```python
# setup_hook(), existing code:
for name in ('status', 'health', 'news', 'vibe', 'stats',
             'channels', 'persona', 'forgetme',
             'reloadknowledge', 'proactive_status'):   # ← add 'snapshot' here
    cmd = getattr(type(self), name, None)
    if cmd is None:
        logger.warning(f"⚠️ Command method not found: {name}")
        continue
    cmd._callback = cmd.callback.__get__(self)
    self.add_command(cmd)
```
A new command that is not in this tuple **does not exist** at runtime. Verify with the startup log line `✅ Registered commands: [...]`.

### 2.5 Existing background loops (do not add more)
`status_update` (5 min) · `refresh_server_intel` (30 min) · `daily_prune` (24 h) · `ollama_watchdog` (5 min). Snapshot adds **zero** loops.

---

## 3. Hard Constraints

### 3.1 MUST
1. **Pull-only.** Work happens inside the `!snapshot` invocation. No listeners, loops, schedulers, or startup work beyond registering the command.
2. **Additive & surgical.** All new logic lives in `src/snapshot/`. `discord_bot.py` changes are limited to: one import, one entry in the `setup_hook` tuple, one command method (≤ 40 lines) that delegates. `agent.py` may receive **one** backward-compatible change (see §7.4). Nothing else moves.
3. **Never crash, never go silent.** The command always replies with *something useful* within the time budget, even if Discord, Ollama, or the network fail mid-way.
4. **Privacy floor (defaults):** aggregates only; no message text leaves the process; no quotes; display names only for the owner and only as "top voices" counts; non-owner views are count-only. Raw message content is discarded immediately after scoring.
5. **Discard, don't store.** No DB writes. In-memory TTL cache only.
6. **Small-model safe.** Works on `qwen3.5:latest` with ≤ ~2,500 input tokens and ≤ 220 output tokens; deterministic fallback when the LLM is slow or absent (§7).
7. **Persona-consistent.** Summary voice = existing bot voice: casual, unisex, "bro energy", **no pet names** ("babe", "baby", "handsome" are banned), no flirty tone, no hedging like "I can't see channels."
8. **Configurable via `.env`** with sane defaults (§8). Every knob documented in `.env.example`.
9. **Logging without content.** One structured INFO line per snapshot (counts, timings, skips). Never log message text, never log the rules channel's content.
10. **Tests.** `pytest` suite that runs with **no Discord or Ollama connection** (§9).

### 3.2 MUST NOT touch (regression list — all verified working today)
| Feature | Where | Status |
|---|---|---|
| Persona / `OWNER_ADDON` / `HELP_ADDON` / `VIBE_ADDON` | `agent.py` PromptConstructor | keep byte-identical |
| `#thinkhard` owner opt-in (`_parse_thinkhard`) | `discord_bot.py` | keep |
| Mention → command dispatch (`_handle_mention`) | `discord_bot.py` | keep |
| DM handling (`_handle_dm`) | `discord_bot.py` | keep |
| `_gather_server_intel` (oldest-first, bot-filtered rules read) | `discord_bot.py` | keep |
| `add_history()` refusing `role='assistant'` (anti-poisoning) | `agent.py` | keep |
| `_human_typing_delay()` formula | `discord_bot.py` | keep, reuse |
| Commands: `status health news vibe stats channels persona forgetme reloadknowledge proactive_status` | both | keep signatures + behavior |
| Proactive engagement scoring | `discord_bot.py` | keep |
| The four `tasks.loop`s | `discord_bot.py` | keep |
| `requirements.txt` runtime deps | root | no new runtime deps (dev deps go in `requirements-dev.txt`) |

---

## 4. Architecture

### 4.1 Package layout (new)
```
src/snapshot/
├── __init__.py        # exports: run_snapshot(), SnapshotOptions
├── options.py         # parse "!snapshot hours:12 detail:high fresh" → SnapshotOptions (pure)
├── collector.py       # Discord I/O: walk channels, fetch history, isolate failures (async)
├── metrics.py         # pure functions: per-channel + server aggregates, highlights
├── summarizer.py      # LLM stage: compact prompt, think-tag stripping, timeout, fallback
├── render.py          # discord.Embed builder with hard limit enforcement + plain-text fallback
└── cache.py           # TTL cache + circuit breaker (pure, no I/O)
```
Rule of thumb: **I/O modules (`collector`, `summarizer`) are thin; logic modules (`options`, `metrics`, `render`, `cache`) are pure and unit-tested.** No module imports `discord_bot.py`.

### 4.2 Data flow
```
!snapshot [opts]
   │
   ▼ options.parse()  ───────────────────────────► SnapshotOptions (validated, clamped)
   │
   ▼ cache.get(key)  ── hit & !fresh ─────────────► render(cached) + "cached Xs ago" footer
   │ miss
   ▼ collector.collect(guild, opts, analyzer, deadline)
   │     for ch in guild.text_channels (Semaphore N):
   │        history(after=window_start, limit=cap, oldest_first=False)
   │        per message: skip bots → analyzer.analyze(content)['compound_score'] → drop content
   │        → ChannelSample (counts, unique authors, compound list, first/last ts, author→count)
   │     failures isolated per channel → ChannelSample(status="skipped", reason=...)
   ▼
   ▼ metrics.compute(samples, opts) ──────────────► ServerSnapshot (sorted channels, totals, highlights)
   │
   ▼ summarizer.summarize(snapshot, ollama, opts, deadline) ─► str | None   (None ⇒ fallback text)
   │
   ▼ render.embed(snapshot, summary, meta) ───────► discord.Embed (or chunked text on failure)
   │
   ▼ cache.put(key, snapshot, summary)
   ▼ ctx.send(...)  (after _human_typing_delay)
```

### 4.3 Core types (dataclasses; keep them small)
```python
@dataclass(frozen=True)
class SnapshotOptions:
    hours: int            # window; clamp 1..168
    detail: str           # "low" | "medium" | "high"
    channels: tuple[str, ...]   # explicit channel names/ids; empty = all
    top: int              # channels shown in embed; clamp 3..25
    fresh: bool           # bypass cache
    include_bots: bool

@dataclass
class ChannelSample:
    channel_id: int; name: str; category: str | None
    status: str           # "ok" | "skipped"
    reason: str | None    # "no_access" | "rate_limited" | "timeout" | "http_error" | "deadline"
    msg_count: int = 0
    unique_authors: int = 0
    compounds: list[float] = field(default_factory=list)
    author_counts: dict[int, int] = field(default_factory=dict)   # id → count (names resolved at render, owner only)
    first_ts: datetime | None = None; last_ts: datetime | None = None
    last_sixth_count: int = 0    # msgs in the final 1/6 of the window (burst detection)

@dataclass
class ChannelMetrics:
    name: str; msg_count: int; unique_authors: int
    avg_compound: float | None; mood: str          # via engine._mood_label
    activity_tier: str                              # "hot" | "warm" | "low" | "quiet" | "skipped"
    burst: bool; top_author_ids: list[int]

@dataclass
class ServerSnapshot:
    guild_name: str; window_hours: int; taken_at: datetime
    channels: list[ChannelMetrics]                  # sorted by msg_count desc
    scanned: int; skipped: int; skipped_reasons: dict[str, int]
    total_msgs: int; total_unique_authors: int
    highlights: list[str]                           # rule-based, ≤ 6
    partial: bool                                   # deadline hit or ≥1 skip
    elapsed_s: float
```

---

## 5. Behavior Spec

### 5.1 Command grammar
```
!snapshot                         all text channels, SNAPSHOT_DEFAULT_HOURS, detail=medium
!snapshot hours:24                custom window (1–168)
!snapshot channels:general,memes  subset by name (with or without #) or id
!snapshot detail:low|medium|high  low = counts only · medium = + mood + highlights · high = + top voices (owner only)
!snapshot top:8                   number of channels listed (3–25)
!snapshot fresh                   bypass the cache
aliases: !snap, !whatsgoingon
```
Unknown tokens are ignored; reply footer lists them as `ignored: foo, bar`. Parsing is pure and unit-tested. Values out of range are **clamped**, not rejected.

### 5.2 Access control
`SNAPSHOT_ACCESS` = `owner` (default) | `staff` (owner + members with `manage_messages`) | `everyone`.
- Unauthorized → one short in-voice refusal (no lecture), logged at INFO without content.
- Non-owner callers always get the **count-only** view regardless of `detail:`; names never render for them.

### 5.3 Collection rules
- Channel set: `guild.text_channels` filtered by `permissions_for(guild.me)` having `view_channel` **and** `read_message_history`. Channels failing this are counted as `skipped:no_access` without an API call.
- Per channel: `history(limit=SNAPSHOT_MAX_MSGS_PER_CHANNEL, after=window_start_utc, oldest_first=False)` where `window_start_utc` is a **timezone-aware UTC** datetime.
- Skip `message.author.bot` unless `include_bots`. Skip empty content (embeds-only, stickers) for sentiment but count them as activity.
- Score with `analyzer.analyze(content)['compound_score']`; **do not keep `content`**.
- Concurrency: `asyncio.Semaphore(SNAPSHOT_CONCURRENCY)`; do not re-implement Discord rate limiting — discord.py handles 429 buckets. Bound concurrency and wall-clock instead.
- Deadline: `SNAPSHOT_MAX_SECONDS` total for collection. Channels not started by the deadline → `skipped:deadline`. In-flight tasks get a short grace then are cancelled.

### 5.4 Metrics (pure; `metrics.py`)
- `avg_compound` = mean of compounds if ≥ `SNAPSHOT_MIN_MSGS_FOR_MOOD` (default 3) else `None` → mood `"n/a"`.
- `mood` label via `engine._mood_label(avg_compound)` for consistency with `!vibe`.
- `activity_tier` thresholds relative to the window: `hot` ≥ 30 msgs/6h-equivalent, `warm` ≥ 10, `low` ≥ 1, `quiet` = 0 (scale linearly with `hours/6`).
- `burst` = `last_sixth_count ≥ 0.4 * msg_count` and `msg_count ≥ 6`.
- Highlights (rule-based, ordered, max 6, each ≤ 90 chars):
  1. hottest channel (`#name — N msgs, U voices`)
  2. bursts (`📈 #name heating up in the last hour`)
  3. sentiment extremes (avg ≥ 0.5 with ≥ 5 msgs → `🔥 #name is hyped`; ≤ −0.3 → `⚠️ #name is tense`)
  4. share of silent channels (`⚫ 31 of 48 channels quiet`)
  5. new voices — authors whose `member.joined_at` is within `SNAPSHOT_NEW_MEMBER_DAYS` (default 7) and who posted in-window (count only for non-owner; names for owner)
  6. skips (`🔒 3 channels skipped (no access)`)

### 5.5 Output (embed)
- **Title:** `📸 NSZ Snapshot — <local time, America/Chicago>`
- **Description:** `Window: last {hours}h · Channels: {scanned} scanned, {skipped} skipped · Messages: {total} · Voices: {unique}` + `⚠️ Partial — {reason}` when `partial`.
- **Field "Channels":** top `N` lines: `{tier_emoji} #{name:<18} {msgs:>4} msgs · {authors:>2} voices · {mood}`; then `+ {k} more quiet` if truncated. Use a code block for alignment.
- **Field "Highlights":** the rule-based list.
- **Field "Read":** the LLM summary (3–6 lines) — or, on fallback, a deterministic 2-line summary (`{hot_count} channels hot, {quiet_count} quiet; mood {overall_mood}`) with a discreet `(AI read unavailable)` tail.
- **Field "Top voices"** (owner + `detail:high` only): `@displayname ×N` for up to 5 authors, resolved from `guild.get_member(id)` at render time; fall back to `member#id` if uncached.
- **Footer:** `took {elapsed:.1f}s · cache {age}s · model {name}` + `ignored: …` if any.
- Enforce Discord limits: field ≤ 1,024 chars, description ≤ 4,096, total ≤ 6,000, ≤ 25 fields. If the embed still fails to send (`HTTPException`), send the plain-text rendering via `_chunk`.
- Tier emojis: hot 🟢 · warm 🟡 · low ⚪ · quiet ⚫ · skipped 🔒

### 5.6 Cache
Key = `(guild_id, hours, detail, channels, include_bots)`. TTL `SNAPSHOT_CACHE_TTL` (default 300 s). Hit ⇒ re-render cached snapshot + summary with `cache {age}s` footer; owner sees a one-line hint `add "fresh" to refetch`. `fresh` bypasses and overwrites.

---

## 6. Resilience Spec ("bulletproof, self-healing")

Every row below is a **required behavior** with a test.

| Failure | Where | Required behavior |
|---|---|---|
| `discord.Forbidden` on `history()` | collector | isolate → `skipped:no_access`; continue |
| `discord.HTTPException` 429 | collector | rely on discord.py bucket handling; if it still surfaces, retry once after `retry_after` (cap 5 s) else `skipped:rate_limited` |
| Other `HTTPException` / `DiscordServerError` | collector | `skipped:http_error`; continue |
| Per-channel timeout (`SNAPSHOT_CHANNEL_TIMEOUT`, default 8 s) | collector | `skipped:timeout`; continue |
| Global deadline reached | collector | stop scheduling; mark rest `skipped:deadline`; `partial=True`; render what exists |
| Zero channels readable | metrics/render | reply with a clear "0 channels readable — check role permissions" embed; still no exception |
| Ollama unreachable / timeout / empty / garbage output | summarizer | return `None` → deterministic fallback text; increment connector failure metric; never block > `SNAPSHOT_LLM_TIMEOUT` |
| LLM emits `<think>…</think>` or markdown fences | summarizer | strip; if remaining text < 20 chars → treat as `None` |
| LLM output violates voice (pet names, "I can't see") | summarizer | regex screen for banned tokens → drop offending line(s); if nothing left → fallback |
| Embed too large / send fails | render | trim fields in order: Top voices → Highlights → Channels tail; then plain-text chunks |
| Member cache miss when resolving names | render | show `user-{id}`; never call `fetch_member` in a loop |
| Unexpected exception anywhere | command method | catch-all → log with traceback (no content) → send one-line apology in voice → **circuit breaker** trip |
| 3 consecutive failed snapshots | cache/breaker | breaker open for `SNAPSHOT_BREAKER_COOLDOWN` (default 60 s): reply "snapshot cooling down, try in Ns"; half-open after cooldown |
| Two `!snapshot` invocations overlapping | command method | per-guild `asyncio.Lock`; second caller gets "one's already cooking" and the first result is delivered to both channels if different, else once |

Non-functional guarantees:
- **Hard wall clock:** collection ≤ `SNAPSHOT_MAX_SECONDS` (default 25) + LLM ≤ `SNAPSHOT_LLM_TIMEOUT` (default 45) ⇒ the user always sees a reply in ≤ ~75 s, typically 5–15 s. Keep the typing indicator alive via `_keep_typing` for the whole span.
- **Memory:** compounds are floats; no content retained; cache holds ≤ `SNAPSHOT_CACHE_MAX_ENTRIES` (default 8) snapshots, LRU-evicted.
- **Cancellation-safe:** all awaits inside `asyncio.wait_for` or guarded by the deadline; tasks cancelled on exit; no orphan coroutines (check `asyncio.all_tasks()` in tests).

---

## 7. Small Local Model Spec — `qwen3.5:latest`

The LLM sees **numbers, not messages.** It writes prose about a table. This keeps the prompt tiny, the privacy floor intact, and the output stable on a 6–7 GB local model.

### 7.1 Model selection
- Use `SNAPSHOT_MODEL` if set, else `OLLAMA_MODEL`. Default target: `qwen3.5:latest`. Must also work unchanged with `minimax-m3:cloud` and `qwen3.5:4b`.
- Log the resolved model name once per snapshot (footer + INFO line).

### 7.2 Prompt contract (keep total input ≤ ~2,500 tokens; the table is ≤ 1,800 chars)
**System (short, standalone — NOT the full chat persona prompt):**
```
You are Acheron, the No Sleep Zone's server bot. Voice: casual, direct, unisex, bro energy.
Never use pet names. Never say you can't see channels — you are reading a real snapshot table.
Write 3–6 short lines, plain text, no headers, no bullet symbols, no emojis, no JSON.
Line 1: overall energy in one sentence. Then: what's hot, what's quiet, any mood swings.
Mention channels as #name. Do not invent channels or numbers not in the table.
```
**User:**
```
SNAPSHOT {guild} · last {hours}h · {scanned} channels · {total} msgs · {unique} voices
name | msgs | voices | mood | burst
#general | 47 | 12 | chill | no
#vibe-check | 31 | 9 | hyped | yes
... (top N rows only; then) +{k} quiet channels
HIGHLIGHTS: {highlights joined by ' | '}
```
**Generation options:** `num_predict = SNAPSHOT_LLM_NUM_PREDICT` (default 220), `temperature = 0.3`, `think = False` when the client/model supports it (see §7.4), `timeout = SNAPSHOT_LLM_TIMEOUT`.

### 7.3 Post-processing (always)
1. Strip `<think>…</think>` (multiline, non-greedy), code fences, leading role labels.
2. Collapse to ≤ 6 non-empty lines; hard-cap 700 chars.
3. Banned-token screen (case-insensitive): `babe, baby, handsome, sweetie, hun, i can't see, i cannot see, i don't have access` → drop those lines.
4. Empty/too-short result → return `None` (renderer uses the deterministic fallback). **Never** surface an error string to Discord as if it were the summary.

### 7.4 The one permitted `agent.py` change (backward-compatible)
Extend `OllamaConnector.generate` with optional keyword args, defaults preserving current behavior:
```python
async def generate(self, messages, num_predict: int = None,
                   temperature: float | None = None,
                   think: bool | None = None,
                   timeout: float | None = None) -> str
```
- `temperature`/`num_predict` merge into the existing `options` dict; `think` is passed to `client.chat(..., think=think)` only when not `None` and **wrapped in try/except TypeError** so older clients/models without the parameter fall back to a plain call.
- Existing callers pass nothing new ⇒ identical behavior. Existing health metrics keep updating.
- Do **not** change retry semantics or the "never raises" contract.

### 7.5 Why this is small-model safe
- No JSON parsing of model output (small models break JSON silently).
- Pre-computed numbers ⇒ the model can't be "wrong about the data", only stylistically off — and style is screened.
- Low temperature, tight token cap, single-shot, no tool calls.
- The whole feature works with the LLM turned off (`SNAPSHOT_LLM_ENABLED=false`) — that path is also what tests exercise.

---

## 8. Configuration (add to `.env.example` with these comments; read via `os.getenv` in `snapshot/options.py` or a tiny `snapshot/config.py`)

```ini
# ── !snapshot (on-demand server awareness; pull-only, no background work) ──
SNAPSHOT_ACCESS=owner                 # owner | staff | everyone
SNAPSHOT_DEFAULT_HOURS=6              # window when hours: not given (1–168)
SNAPSHOT_DEFAULT_DETAIL=medium        # low | medium | high
SNAPSHOT_TOP_CHANNELS=12              # channels listed in the embed (3–25)
SNAPSHOT_MAX_MSGS_PER_CHANNEL=60      # history() limit per channel
SNAPSHOT_CONCURRENCY=4                # parallel channel fetches
SNAPSHOT_CHANNEL_TIMEOUT=8            # seconds per channel
SNAPSHOT_MAX_SECONDS=25               # collection deadline (partial results after this)
SNAPSHOT_INCLUDE_BOTS=false           # count bot messages? (default excludes self-poisoning)
SNAPSHOT_MIN_MSGS_FOR_MOOD=3          # below this, mood shows n/a
SNAPSHOT_NEW_MEMBER_DAYS=7            # "new voice" threshold
SNAPSHOT_CACHE_TTL=300                # seconds; "fresh" bypasses
SNAPSHOT_CACHE_MAX_ENTRIES=8
SNAPSHOT_BREAKER_COOLDOWN=60          # seconds after 3 consecutive failures
SNAPSHOT_LLM_ENABLED=true             # false ⇒ deterministic summary only
SNAPSHOT_MODEL=qwen3.5:latest         # overrides OLLAMA_MODEL for the summary stage only
SNAPSHOT_LLM_NUM_PREDICT=220
SNAPSHOT_LLM_TIMEOUT=45               # seconds
```
All values validated + clamped at load; a bad value logs a WARNING and uses the default — never raises at import time.

---

## 9. Tests & Acceptance

### 9.1 Test layout
```
tests/
├── conftest.py            # fakes: FakeGuild, FakeChannel(history=…, raises=…), FakeMember, FakeOllama
├── test_options.py        # grammar, clamping, ignored tokens, aliases
├── test_metrics.py        # tiers, mood gating, burst, highlights ordering, zero-channel case
├── test_collector.py      # Forbidden/429/timeout/deadline isolation; bots skipped; content not retained
├── test_summarizer.py     # think-tag strip, banned-token screen, timeout ⇒ None, disabled ⇒ None
├── test_render.py         # embed limits, trimming order, plain-text fallback, non-owner never sees names
└── test_cache.py          # TTL, LRU, fresh bypass, breaker trip/half-open
requirements-dev.txt       # pytest, pytest-asyncio  (runtime requirements.txt unchanged)
```
Run: `venv/bin/pip install -r requirements-dev.txt && venv/bin/pytest -q` — must pass offline.

### 9.2 Acceptance criteria (all required)
- [ ] Startup log shows `snapshot` in `✅ Registered commands: [...]`; all 10 pre-existing commands still listed.
- [ ] `!snapshot` from the owner in a channel returns one embed in ≤ 30 s on a 48-channel server with ~6 h window.
- [ ] `!snapshot` from a non-owner with `SNAPSHOT_ACCESS=owner` → in-voice refusal; with `everyone` → count-only embed, no names.
- [ ] Revoke the bot's `read_message_history` on one channel → that channel appears as `skipped (no_access)`; everything else renders; `partial` banner shows.
- [ ] Stop Ollama (or set a bogus `SNAPSHOT_MODEL`) → embed still arrives with the deterministic "Read" fallback within the LLM timeout; no traceback reaches Discord.
- [ ] Set `SNAPSHOT_MODEL=qwen3.5:latest` → summary has no `<think>` residue, ≤ 6 lines, no banned tokens, references only real channel names.
- [ ] `!snapshot` twice within 5 min → second reply footer shows `cache Ns`; `!snapshot fresh` re-collects.
- [ ] `grep -n "tasks.loop" src/discord_bot.py` count unchanged (4). `git diff --stat` shows `agent.py` touched only in `OllamaConnector.generate`.
- [ ] `!vibe !stats !health !persona !channels !news !status` produce the same output shape as before (spot check).
- [ ] No message content in `logs/daddyclintbot.log` for a snapshot run (grep a known phrase posted during the test).
- [ ] `pytest -q` green offline.

### 9.3 Manual verification script (for the operator)
```bash
cd ~/projects/daddyclintbot
# stop the running instance first — one gateway connection per token
pkill -TERM -f "src/discord_bot.py"; sleep 3
venv/bin/pytest -q
venv/bin/python src/discord_bot.py            # .env supplies token/host; watch for "Registered commands"
# in Discord: !snapshot · !snapshot hours:24 detail:high · !snapshot fresh · (non-owner) !snapshot
```

---

## 10. Deliverables & Git Hygiene

1. `src/snapshot/` package (7 files) + `tests/` (7 files) + `requirements-dev.txt`
2. `src/discord_bot.py`: import, tuple entry, one `snapshot` command method (aliases `snap`, `whatsgoingon`)
3. `src/agent.py`: `OllamaConnector.generate` optional kwargs only
4. `.env.example`: §8 block appended
5. `README.md`: one short "!snapshot" section (usage + privacy note); `SETUP_NOTES.md`: env knobs pointer
6. Commit as **one** feature commit on top of `main` (do not squash unrelated pending fixes into it):
   `feat(snapshot): on-demand server awareness command with resilient collection and local-model summary`
7. Push over SSH (`git@github.com:daddyclint82/daddyclintbot.git`). **Never** prompt the operator for a GitHub username/password — password auth is dead; SSH key + `gh` are already configured on the host.

### Definition of Done
All §9.2 boxes checked, `pytest` green, the operator has seen one real `!snapshot` embed produced with `qwen3.5:latest`, and `git log -1` shows the feature commit with `agent.py`'s diff confined to `generate()`.

---

## Appendix A — Kickoff prompt (paste to the implementing model)

```
You are implementing a feature in the repo daddyclint82/daddyclintbot (Python 3.12, discord.py 2.7.1, ollama 0.6.2).
Read docs/SNAPSHOT_ENGINEERING_BRIEF.md in full and implement Sections 4–10 exactly.

Non-negotiables:
- Pull-only: no listeners, no tasks.loop, no schedulers. Work happens only inside the !snapshot command.
- Additive: new package src/snapshot/; discord_bot.py gets one import, one tuple entry in setup_hook, one ≤40-line command method. agent.py changes only OllamaConnector.generate with optional backward-compatible kwargs.
- Do not modify anything in Section 3.2. Do not add runtime dependencies.
- Every failure in Section 6 degrades gracefully and has a test. The bot never crashes and never replies with nothing.
- Privacy: aggregate only; message content is scored and discarded; no quotes; names only for the owner.
- The LLM stage must run on qwen3.5:latest: numbers-in/prose-out, ≤2,500 input tokens, num_predict ≤220, temperature 0.3, strip <think> tags, banned-token screen, deterministic fallback when the model is unavailable. Voice: casual, unisex, no pet names, no "I can't see channels".
- Follow the discord.py 2.7.1 registration pattern in Section 2.4 or the command will not exist at runtime.

Deliver: a single reviewable diff, the offline `pytest -q` output, and a short note on anything in the brief you could not satisfy and why. Do not invent APIs; where unsure about a discord.py/ollama signature, check the installed version in venv/.
```

## Appendix B — Gotchas learned the hard way (2026-09-04)
- `@commands.command` on a `Bot` subclass body does **not** auto-register; bind `cmd._callback = cmd.callback.__get__(self)` before `add_command`.
- Two processes with one bot token ⇒ the second stalls silently at gateway connect. Kill the first.
- `discord.Intents` attribute is `presences` (plural); `message_content` and `members` are already enabled in the Developer Portal for this app.
- `channel.history(after=…)` needs a **timezone-aware UTC** datetime; with `after` set, pass `oldest_first=False` explicitly to get the most recent messages.
- Qwen3-family models emit `<think>…</think>`; strip it even if you pass `think=False`.
- Reading a channel's most recent messages while the bot is also posting there **feeds the bot its own output**. That is why bots are excluded by default and content is never stored.
- Discord embed limits: 256 title · 4,096 description · 1,024 per field value · 25 fields · 6,000 total.
