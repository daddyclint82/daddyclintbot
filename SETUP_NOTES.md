# DaddyClintBot - Setup & Configuration Notes

## Project Overview
Discord bot with psychological engagement engine using Ollama LLM and VADER sentiment analysis.

## Critical Environment Setup

### Required Environment Variables

Create `.env` file in project root:

```bash
# Discord Bot Configuration
DISCORD_TOKEN=your_discord_token_here
OLLAMA_MODEL=phi3
OLLAMA_HOST=http://172.18.224.1:11434

# Bot Settings
BOT_PREFIX=!
BOT_NAME=DaddyClintBot

# Database
DB_PATH=data/daddyclintbot.db

# Response Timing
MIN_DELAY=1.0
MAX_DELAY=15.0

# Logging
LOG_LEVEL=INFO
```

### CRITICAL: Ollama Host Configuration

**For WSL/Remote Ollama instances**, you MUST export this before starting:

```bash
export OLLAMA_HOST="http://172.18.224.1:11434"
```

This IP (172.18.224.1) is the Windows host IP from WSL2. If this changes, update accordingly.

## Installation Steps

### 1. Navigate to Project
```bash
cd /home/daddyclint82/.openclaw/workspace/daddyclintbot
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your actual DISCORD_TOKEN and settings
nano .env
```

### 5. Start Ollama (in separate terminal or background)
```bash
ollama run phi3
# Or: ollama run llama3
# Or: ollama run mistral
```

### 6. Export Ollama Host (CRITICAL)
```bash
export OLLAMA_HOST="http://172.18.224.1:11434"
```

### 7. Start the Bot
```bash
python src/discord_bot.py
```

## Bot Commands

- `!status` - Check bot and psychological engine status
- `!persona` - Display bot persona information

## Interaction Methods

1. **Direct Message (DM)** - Full psychological engagement
2. **@mention in channels** - Quick responses

## Key Technical Fixes Applied

### Fix 1: Async Ollama Integration
**Problem:** Ollama's synchronous `chat()` call was blocking Discord's event loop for 30+ seconds, causing disconnections.

**Solution:** Modified `OllamaConnector` class to use `ThreadPoolExecutor` with `asyncio.run_in_executor()`:
```python
self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
response = await loop.run_in_executor(
    self.executor,
    partial(ollama.chat, ...)
)
```

**File:** `src/agent.py`

### Fix 2: Discord Typing Indicator
**Problem:** Discord disconnecting during long LLM generation.

**Solution:** Created `_keep_typing()` background task that refreshes typing indicator every 5 seconds while processing:
```python
async def _keep_typing(self, channel):
    while True:
        await channel.trigger_typing()
        await asyncio.sleep(5)
```

**File:** `src/discord_bot.py`

### Fix 3: Removed Artificial Delay for Discord
**Problem:** Combined Ollama generation time + artificial delay = Discord timeout.

**Solution:** Created `_process_discord_message()` method that skips the `asyncio.sleep()` delay used in CLI mode:
```python
# In CLI mode: has delay
# In Discord mode: no delay, only Ollama generation time
```

**File:** `src/discord_bot.py`

## Project Structure

```
daddyclintbot/
├── src/
│   ├── agent.py              # Psychological engine (DaddyClintBot class)
│   └── discord_bot.py        # Discord integration
├── data/                     # SQLite database (auto-created)
├── logs/                     # Log files (auto-created)
│   ├── daddyclintbot.log     # Main bot logs
│   └── discord_bot.log       # Discord-specific logs
├── .env                      # Environment variables (NOT in git)
├── .env.example              # Template for .env
└── requirements.txt          # Python dependencies
```

## Troubleshooting

### Issue: "DISCORD_TOKEN not found"
**Fix:** Ensure `.env` file exists and contains valid token

### Issue: Ollama connection fails
**Fix:** 
1. Check Ollama is running: `ollama list`
2. Export correct host: `export OLLAMA_HOST="http://172.18.224.1:11434"`
3. Verify IP matches your Windows host from WSL2

### Issue: "heartbeat blocked" warnings
**Fix:** This was resolved by making Ollama calls async. If it persists:
1. Kill bot: `pkill -f discord_bot.py`
2. Restart Ollama
3. Restart bot

### Issue: Bot responds slowly
**Expected:** Ollama generation takes 10-30 seconds on CPU. This is normal.

## Model Switching

To switch LLM models:

1. Edit `.env`:
```bash
OLLAMA_MODEL=llama3
# or
OLLAMA_MODEL=mistral
```

2. Pull model if needed:
```bash
ollama pull llama3
```

3. Restart bot

## Last Updated
2026-05-12 - Initial deployment with Ollama integration fixes

## Next Steps / TODO
- [ ] Add more Discord slash commands
- [ ] Implement channel-specific responses
- [ ] Add user memory persistence across sessions
- [ ] Create dashboard for monitoring engagement metrics
- [ ] Deploy to production server (not WSL)

---
**Remember:** Always export `OLLAMA_HOST` before starting the bot!
