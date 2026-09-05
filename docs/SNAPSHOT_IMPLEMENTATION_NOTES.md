# `!snapshot` — Implementation Notes

What was delivered for the snapshot engineering brief, where the
implementation deliberately deviates from the brief (and why), and what
still needs verification on the live server.

Delivered as a single feature commit: `dd45afe` on `main`.
Test status at delivery: **71/71 passing, fully offline** (no Discord
connection, no Ollama server needed).

---

## What was done

### New `src/snapshot/` package (7 modules, I/O thin / logic pure)

| Module | Responsibility |
|---|---|
| `options.py` | Arg grammar parser (`hours: channels: detail: top: fresh`) + `SNAPSHOT_*` env config. Clamps everything, never rejects, lists ignored tokens. No Discord imports. |
| `collector.py` | The only module that touches Discord. Semaphore-bounded channel walk, per-channel timeout, global deadline with partial results, 429 retry-once, Forbidden pre-check (no API call wasted), per-channel failure isolation. **Scores content in-loop and discards it immediately.** |
| `metrics.py` | Tiers (hot/warm/low/quiet, scaled to window), mood labels, burst detection, highlights, server totals. Pure functions. |
| `summarizer.py` | Numbers-table prompt → one Ollama call → aggressive post-processing (`<think>`/fence stripping, role-label stripping, banned-token screen, 6-line cap) → deterministic fallback line on any failure. Returns `None` on garbage; never raises. |
| `render.py` | Single embed with hard Discord limits enforced (25 fields / 1,024 per field / 6,000 total), trim order: Top voices → Highlights → Read → Channels. Always builds a plain-text fallback. |
| `cache.py` | TTL + LRU cache (5 min / 32 entries) and a circuit breaker (3 failures → 60 s cooldown → half-open). |
| `__init__.py` | `run_snapshot()` orchestrator: access gate → parse → cache → breaker → in-flight → collect → metrics → summarize → render. Catches everything; never leaves a Discord request silent. |

### Command surface (`src/discord_bot.py`)

- `!snapshot` with aliases `!snap`, `!whatsgoingon` — decorated method plus
  explicit registration in `setup_hook` (see deviation #1).
- Owner-only by default. `SNAPSHOT_ACCESS=staff` allows members with
  Manage Messages; `SNAPSHOT_ACCESS=everyone` allows everyone (count-only
  for non-owners).
- Non-owners are **always** count-only: no Top voices, no Highlights, no
  member names — regardless of the `detail:` they ask for.
- Three-line addition to the file header comment.

### `src/agent.py` — additive only

`OllamaConnector.generate()` gained optional `temperature`, `think`,
`timeout`, and `model` kwargs (snapshot needs cold/cheap/kay'o-tuned
calls; existing hot/chatty behavior is the unchanged default). The retry
loop and never-raises contract are untouched. Verified: the diff is two
hunks, both inside `generate()`.

### Tests (`tests/`, 71 passing)

Fakes for Guild/Channel/Message/Member/Connector; real `PsychologicalAnalyzer`
against `vaderSentiment`. Covers: grammar/clamps/ignored tokens, tier scaling,
mood gating, burst, highlights ordering, zero-channel case, per-failure
isolation (Forbidden/HTTP/429-retry/timeout/deadline), bots excluded by
default, **content-never-retained**, think-tag/banned-token/garbage handling,
embed hard limits and trim order, non-owner name-hiding, cache TTL/LRU,
breaker trip/half-open, access gate, cache hit/fresh bypass, LLM-off and
LLM-down still delivering, never-silent on internal explosion, in-flight
guard, zero-readable-channels case.

---

## What could not be done exactly as briefed (and why)

1. **§2.4's registration tuple did not exist in GitHub HEAD.** The brief's
   "verified live" table claimed commands auto-register and that no
   registration step was needed; on discord.py 2.7.1 that is false —
   class-body `@commands.command()` decorators in a `commands.Bot`
   subclass do **not** self-register, so HEAD's ten pre-existing commands
   were dead. The fix went in as explicit registration for all 11
   commands in `setup_hook`, plus a `✅ Registered commands` startup log.
   **Consequence:** the live WSL copy contains uncommitted changes (the
   registration tuple, `add_history` anti-poisoning, the bot-filtered
   rules read) that HEAD lacks. Pulling `main` will likely surface a
   small merge around `setup_hook` — keep the local versions of those
   fixes; they are compatible with the new snapshot wiring.

2. **§7.1 vs §7.4 conflict.** "Reuse `OllamaConnector` exactly as-is" and
   "`SNAPSHOT_MODEL` override" cannot both hold — the model to use is a
   per-call decision. Resolved by adding an optional `model` kwarg to
   `generate()` alongside the other optional kwargs. Still confined to
   `generate()`; still backward-compatible.

3. **`ChannelMetrics.top_authors` stores `(author_id, count)` tuples**
   instead of the brief's `top_author_ids: list[int]`. The high-detail
   "Top voices" field renders `@name ×N`, which needs counts. IDs-only
   would have forced a second pass just to recount.

4. **`ServerSnapshot` gained an `overall_mood` field** (not in the brief's
   sketch). The deterministic fallback line when the LLM is off/down is
   `"No AI read this run — mood {mood}, {n} msgs across {c} channels"`
   and needs the aggregate mood computed once, not re-derived in render.

5. **The `channels:` parser tolerates spaces after commas**
   (`channels:general, memes` works). Discord users will type the space;
   the naive whitespace tokenizer would have silently dropped everything
   after the first comma. Continuation tokens are consumed while the value
   ends with `,` or the next token starts with one.

6. **In-flight rule interpretation:** identical concurrent requests share
   the running result; requests with *different* args get "one's already
   cooking — try again shortly." Sharing a 6-hour-window result with a
   `hours:24` caller would deliver the wrong window.

7. **Test layout:** the brief listed six test files; a seventh
   (`tests/test_run_snapshot.py`) was added for orchestration coverage
   (access gate, cache, breaker, in-flight, never-silent). Better split
   than stuffing orchestration into `test_render.py`.

---

## Acceptance criteria status

Verified offline in this session:

- ✅ `!snapshot` registers (incl. `snap` / `whatsgoingon` aliases) and all
  10 pre-existing commands register alongside it
- ✅ `git diff HEAD -- src/agent.py` touches only `generate()`
- ✅ `grep -c "tasks.loop" src/discord_bot.py` still reports 4
- ✅ `pytest -q` — 71 passed
- ✅ No code path logs or stores message content (collector discards
  content in-loop; samples carry counts/compounds/IDs only)

Requires the live server (cannot be proven offline):

- ⏳ Real `!snapshot` returns one embed across all 48 channels within ~30 s
- ⏳ Non-owner receives the in-voice refusal (default `SNAPSHOT_ACCESS=owner`)
- ⏳ A channel with revoked Read Message History appears in the
  `⚠️ Partial …` banner without an API error
- ⏳ `SNAPSHOT_MODEL=qwen3.5:latest` produces a clean 3–6 line read
  (otherwise tighten `SNAPSHOT_LLM_NUM_PREDICT`)
- ⏳ Second call within 5 min shows `cache Ns` in the footer

---

## Runbook (first live run)

```bash
git pull              # resolve the setup_hook merge — keep your local fixes
pkill -TERM -f "src/discord_bot.py"   # one token = one gateway connection
venv/bin/pytest -q    # 71 passed expected, needs requirements-dev.txt
python src/discord_bot.py
```

Then in Discord: `!snapshot`, `!snapshot hours:24 detail:high fresh`,
and a non-owner account trying `!snapshot` (expect the refusal).

Optional knob for the read quality: `SNAPSHOT_MODEL=qwen3.5:latest` in
`.env` (uses the existing Ollama path — **not** the large think-mode
`#thinkhard` model).
