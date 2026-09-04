# DaddyClintBot

## Status
✅ Deployed and Running via systemd

## Quick Links
- **Location:** `/home/daddyclint82/.openclaw/workspace/daddyclintbot/`
- **Started:** 2026-05-12

## Description

Automated Discord bot for text-based engagement and community moderation.

## Architecture

### System Overview

Automated Discord engagement and moderation bot with proactive chat capabilities and scheduled tasks.

### Components

| Component | Purpose |
|-----------|---------|
| **Discord Client** | Connects to Discord Gateway API. Listens for messages, reactions, and guild events. |
| **Engagement Engine** | Proactive chat responses triggered by keywords/patterns. Community moderation rules enforcement. |
| **Task Scheduler** | Auto-scheduled recurring tasks (messages, cleanups, etc.). Configurable intervals and triggers. |
| **Data Persistence** | SQLite database for user data, logs, and configuration. Local file storage — no external DB dependency. |

### Data Flow

1. Discord event triggers (message, reaction, join, etc.)
2. Bot evaluates against engagement rules
3. Response or moderation action executed
4. Activity logged to SQLite

### Key Design Decisions

- **Systemd service:** Runs as persistent background service.
- **SQLite:** Zero-config local persistence.
- **Proactive responses:** Not just command-driven — engages naturally.

### Tech Stack

- Node.js / discord.js
- SQLite (data persistence)
- Systemd (service management)

## !snapshot — On-Demand Server Awareness

Pull-only server snapshot (no background monitoring): walks every readable text
channel, computes per-channel activity + sentiment aggregates in-process, and
returns one embed with an AI-written read of the room.

```
!snapshot                      all channels, last 6h
!snapshot hours:24             custom window (1–168)
!snapshot channels:general     subset
!snapshot detail:high          + top voices (owner only)
!snap fresh                    bypass the 5-min cache
```

**Privacy:** aggregates only. Message content is scored and immediately discarded
— never stored, never logged. Member names appear only for the owner.
Owner-only by default (`SNAPSHOT_ACCESS`). All knobs in `.env.example`.

## Getting Started

```bash
# Start the bot
sudo systemctl start daddyclintbot

# Check status
sudo systemctl status daddyclintbot
```

---

*Project details maintained by Clint. Last updated: 2026-05-16*
