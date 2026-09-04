"""LLM stage for !snapshot: the model sees numbers, never messages.

Single-shot, low temperature, tight token cap. Any failure → None, and the
renderer falls back to a deterministic summary. Banned-tone tokens are
screened out line-by-line; <think> residue is stripped even when the model
was told not to think.
"""

import logging
import re
import time
from typing import List, Optional

from .metrics import ServerSnapshot
from .options import SnapshotConfig, SnapshotOptions

logger = logging.getLogger('snapshot.summarizer')

BANNED_TOKENS = (
    'babe', 'baby', 'handsome', 'sweetie', 'hun',
    "i can't see", 'i cannot see', "i don't have access",
)
_BANNED_RE = re.compile('|'.join(re.escape(t) for t in BANNED_TOKENS), re.IGNORECASE)
_THINK_RE = re.compile(r'<think>.*?</think>', re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r'```(?:\w+)?\n?|```')
_LABEL_RE = re.compile(r'^\s*(acheron|assistant|bot|summary)\s*:\s*', re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are Acheron, the No Sleep Zone's server bot. Voice: casual, direct, unisex, bro energy.\n"
    "Never use pet names. Never say you can't see channels — you are reading a real snapshot table.\n"
    "Write 3–6 short lines, plain text, no headers, no bullet symbols, no emojis, no JSON.\n"
    "Line 1: overall energy in one sentence. Then: what's hot, what's quiet, any mood swings.\n"
    "Mention channels as #name. Do not invent channels or numbers not in the table."
)


def build_prompt(snapshot: ServerSnapshot, opts: SnapshotOptions) -> List[dict]:
    """Compact numbers-only prompt (table ≤ ~1,800 chars)."""
    rows = ["name | msgs | voices | mood | burst"]
    shown = [c for c in snapshot.channels
             if c.activity_tier != 'skipped'][:opts.top]
    for c in shown:
        rows.append(f"#{c.name} | {c.msg_count} | {c.unique_authors} | "
                    f"{c.mood} | {'yes' if c.burst else 'no'}")
    quiet_not_shown = sum(1 for c in snapshot.channels
                          if c.activity_tier == 'quiet') - \
        sum(1 for c in shown if c.activity_tier == 'quiet')
    if quiet_not_shown > 0:
        rows.append(f"+{quiet_not_shown} quiet channels")

    table = "\n".join(rows)[:1800]
    highlights = " | ".join(snapshot.highlights) if snapshot.highlights else "none"

    user = (
        f"SNAPSHOT {snapshot.guild_name} · last {snapshot.window_hours}h · "
        f"{snapshot.scanned} channels · {snapshot.total_msgs} msgs · "
        f"{snapshot.total_unique_authors} voices\n"
        f"{table}\n"
        f"HIGHLIGHTS: {highlights}"
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]


def postprocess(text: str) -> Optional[str]:
    """Strip think-tags/fences/labels, screen banned tokens, cap shape."""
    text = _THINK_RE.sub('', text or '')
    text = _FENCE_RE.sub('', text)

    lines = []
    for line in text.splitlines():
        line = _LABEL_RE.sub('', line).strip()
        if not line:
            continue
        if _BANNED_RE.search(line):
            continue  # drop off-voice lines entirely
        lines.append(line)

    result = "\n".join(lines[:6])[:700].strip()
    if len(result) < 20:
        return None
    return result


async def summarize(snapshot: ServerSnapshot, opts: SnapshotOptions,
                    cfg: SnapshotConfig, connector) -> Optional[str]:
    """Return the LLM summary, or None (renderer uses the fallback).

    The connector's 'never raises' contract returns an in-character fallback
    string on total failure — those are detected and treated as None so they
    never surface as if they were the summary.
    """
    if not cfg.llm_enabled:
        logger.info("📸 snapshot LLM: disabled (deterministic fallback)")
        return None

    started = time.monotonic()
    model = cfg.llm_model or connector.model
    try:
        raw = await connector.generate(
            build_prompt(snapshot, opts),
            num_predict=cfg.llm_num_predict,
            temperature=0.3,
            think=False,
            timeout=cfg.llm_timeout,
        )
    except Exception as e:  # noqa: BLE001 — belt and suspenders
        logger.warning(f"⚠️ snapshot LLM raised: {type(e).__name__}")
        return None

    if not raw or raw.strip() in getattr(connector, 'FALLBACKS', ()):  # connector failure text
        logger.info(f"📸 snapshot LLM: unavailable after {time.monotonic() - started:.1f}s "
                    f"(model {model}) → fallback")
        return None

    summary = postprocess(raw)
    logger.info(f"📸 snapshot LLM: {'ok' if summary else 'fallback'} "
                f"in {time.monotonic() - started:.1f}s (model {model})")
    return summary
