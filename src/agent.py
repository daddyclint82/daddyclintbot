"""DaddyClintBot - psychological engagement engine + server guide.

Architecture:
    DatabaseManager       - SQLite (WAL) memory: traits, topics, history, channel activity
    PsychologicalAnalyzer - VADER sentiment + vulnerability flags
    IntentRouter          - classifies messages: chat / help / news
    OllamaConnector       - hardened async LLM client (retries, timeout, small-model tuning)
    PromptConstructor     - compact, small-model-friendly prompts per intent
    ResponseHumanizer     - texting-style post-processing
    DaddyClintBot         - orchestrator
"""

import asyncio
import concurrent.futures
import logging
import logging.handlers
import os
import random
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import ollama

# Load environment variables
load_dotenv()

# Configure logging with rotation so long-running bots don't fill the disk
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/daddyclintbot.log', maxBytes=5_000_000, backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DaddyClintBot')


class DatabaseManager:
    """SQLite database manager for state and memory. WAL mode for crash safety."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv('DB_PATH', 'data/daddyclintbot.db')
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=15000')
        return conn

    def init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ConversationState (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    is_unresolved BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS UserTraits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    trait_key TEXT NOT NULL,
                    trait_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, trait_key)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sentiment_score REAL,
                    response_time REAL,
                    bot_delay REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Short-term conversation context for coherent small-model chats
            conn.execute('''
                CREATE TABLE IF NOT EXISTS History (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Passive channel awareness: what happened where, for news/digests
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ChannelActivity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_history_user ON History(user_id, id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_activity_time ON ChannelActivity(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_user ON Messages(user_id, id)')
            conn.commit()

    # --- Topics (Zeigarnik effect) ---

    def get_unresolved_topics(self, user_id: str, limit: int = 1) -> List[str]:
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT topic FROM ConversationState
                WHERE user_id = ? AND is_unresolved = 1
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
            return [row[0] for row in cursor.fetchall()]

    def set_topic_resolved(self, user_id: str, topic: str):
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE ConversationState
                SET is_unresolved = 0, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND topic = ?
            ''', (user_id, topic))
            conn.commit()

    def add_topic(self, user_id: str, topic: str, is_unresolved: bool = True):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO ConversationState (user_id, topic, is_unresolved, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, topic, is_unresolved))
            conn.commit()

    # --- User traits ---

    def get_user_traits(self, user_id: str, limit: int = 3) -> Dict[str, str]:
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT trait_key, trait_value FROM UserTraits
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            return dict(cursor.fetchall())

    def add_user_trait(self, user_id: str, key: str, value: str):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO UserTraits (user_id, trait_key, trait_value)
                VALUES (?, ?, ?)
            ''', (user_id, key, value))
            conn.commit()

    # --- Messages log ---

    def get_last_message_time(self, user_id: str) -> Optional[datetime]:
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT created_at FROM Messages
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            if result:
                return datetime.fromisoformat(result[0])
            return None

    def log_message(self, user_id: str, content: str, sentiment_score: float,
                    response_time: float, bot_delay: float):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO Messages (user_id, content, sentiment_score, response_time, bot_delay)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, content, sentiment_score, response_time, bot_delay))
            conn.commit()

    # --- Conversation history (small-model context) ---

    def add_history(self, user_id: str, role: str, content: str, max_messages: int = 16):
        """Append a message to the user's history and keep it bounded."""
        with self.get_connection() as conn:
            conn.execute(
                'INSERT INTO History (user_id, role, content) VALUES (?, ?, ?)',
                (user_id, role, content[:800])
            )
            # Keep only the newest max_messages per user
            conn.execute('''
                DELETE FROM History WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM History WHERE user_id = ? ORDER BY id DESC LIMIT ?
                )
            ''', (user_id, user_id, max_messages))
            conn.commit()

    def get_history(self, user_id: str, limit: int = 8) -> List[Dict[str, str]]:
        """Return recent history in chronological order as chat messages."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT role, content FROM (
                    SELECT id, role, content FROM History
                    WHERE user_id = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
            ''', (user_id, limit))
            return [{'role': r, 'content': c} for r, c in cursor.fetchall()]

    # --- Channel activity (news / awareness) ---

    def log_channel_activity(self, channel_id: str, channel_name: str,
                             author: str, content: str):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO ChannelActivity (channel_id, channel_name, author, content)
                VALUES (?, ?, ?, ?)
            ''', (channel_id, channel_name, author, content[:300]))
            conn.commit()

    def get_recent_activity(self, hours: int = 24,
                            exclude_channels: List[str] = None,
                            limit: int = 300) -> List[Tuple[str, str, str, str]]:
        """Recent (channel_name, author, content, created_at), oldest first."""
        exclude_channels = exclude_channels or []
        placeholders = ','.join('?' for _ in exclude_channels) or "''"
        query = f'''
            SELECT channel_name, author, content, created_at FROM (
                SELECT * FROM ChannelActivity
                WHERE created_at >= datetime('now', ?)
                AND channel_name NOT IN ({placeholders})
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
        '''
        with self.get_connection() as conn:
            cursor = conn.execute(
                query, (f'-{hours} hours', *exclude_channels, limit)
            )
            return cursor.fetchall()

    def count_activity(self, hours: int = 24) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ChannelActivity WHERE created_at >= datetime('now', ?)",
                (f'-{hours} hours',)
            )
            return cursor.fetchone()[0]

    def get_channel_stats(self, hours: int = 24,
                          exclude_channels: List[str] = None
                          ) -> List[Tuple[str, int, int]]:
        """Per-channel (name, message_count, distinct_authors), busiest first."""
        exclude_channels = exclude_channels or []
        placeholders = ','.join('?' for _ in exclude_channels) or "''"
        with self.get_connection() as conn:
            cursor = conn.execute(f'''
                SELECT channel_name, COUNT(*) AS n, COUNT(DISTINCT author) AS authors
                FROM ChannelActivity
                WHERE created_at >= datetime('now', ?)
                AND channel_name NOT IN ({placeholders})
                GROUP BY channel_name
                ORDER BY n DESC
            ''', (f'-{hours} hours', *exclude_channels))
            return cursor.fetchall()

    def get_top_contributors(self, hours: int = 24,
                             exclude_channels: List[str] = None,
                             limit: int = 5) -> List[Tuple[str, int]]:
        """Most active authors (join events excluded)."""
        exclude_channels = exclude_channels or []
        placeholders = ','.join('?' for _ in exclude_channels) or "''"
        with self.get_connection() as conn:
            cursor = conn.execute(f'''
                SELECT author, COUNT(*) AS n
                FROM ChannelActivity
                WHERE created_at >= datetime('now', ?)
                AND channel_name NOT IN ({placeholders})
                AND content != '[JOIN]'
                GROUP BY author
                ORDER BY n DESC
                LIMIT ?
            ''', (f'-{hours} hours', *exclude_channels, limit))
            return cursor.fetchall()

    def get_join_count(self, hours: int = 24) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ChannelActivity "
                "WHERE content = '[JOIN]' AND created_at >= datetime('now', ?)",
                (f'-{hours} hours',)
            )
            return cursor.fetchone()[0]

    # --- Privacy & maintenance ---

    def clear_user_data(self, user_id: str):
        """Forget everything about a user (their request)."""
        with self.get_connection() as conn:
            conn.execute('DELETE FROM UserTraits WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM ConversationState WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM History WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM Messages WHERE user_id = ?', (user_id,))
            conn.commit()

    def prune_old_data(self, activity_hours: int = 72, message_days: int = 30):
        """Bound DB growth so the bot can run for months unattended."""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM ChannelActivity WHERE created_at < datetime('now', ?)",
                (f'-{activity_hours} hours',)
            )
            conn.execute(
                "DELETE FROM Messages WHERE created_at < datetime('now', ?)",
                (f'-{message_days} days',)
            )
            conn.commit()
        logger.info("🧹 Database pruned")


class PsychologicalAnalyzer:
    """Lightweight NLP analysis using VADER sentiment"""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        acr_trigger = compound > 0.7

        disclosure_markers = ['i feel', 'i think', 'my', 'i am', "i'm", 'personally']
        vulnerability_score = sum(1 for marker in disclosure_markers
                                  if marker in text.lower()) / len(disclosure_markers)

        return {
            'compound_score': compound,
            'acr_trigger': acr_trigger,
            'vulnerability_score': vulnerability_score,
            'positivity': scores['pos'],
            'negativity': scores['neg'],
            'neutrality': scores['neu']
        }


class LatencyCalculator:
    """Calculate Poisson-based response delays (CLI mode only)"""

    def __init__(self):
        self.min_delay = float(os.getenv('MIN_DELAY', '1.0'))
        self.max_delay = float(os.getenv('MAX_DELAY', '15.0'))

    def calculate_delay(self, user_response_time: float) -> float:
        lambda_param = max(user_response_time * 1.1, 1.0)
        delay = np.random.poisson(lambda_param)
        return min(max(delay, self.min_delay), self.max_delay)


class IntentRouter:
    """Classify what the user wants so the right prompt/knowledge is used."""

    HELP_PATTERNS = [
        r'\bwhere\b.*\b(channel|find|post|go|do i)\b',
        r'\bhow (do|can) i\b',
        r'\bwhat channel\b',
        r'\bwhich channel\b',
        r'\bhow to (join|get|get the|verify|register|sign up)\b',
        r'\b(rules?|guidelines?)\b',
        r'\bget (a|the|that) role\b',
        r'\broles?\b.*\b(get|how|assign)\b',
        r'\bnavigate\b',
        r'\bwhere.*\b(rules?|announcements?|events?)\b',
        r'\bwhat is this (server|place)\b',
        r'\bwho (is|are) the (admin|mod|owner)\b',
    ]
    NEWS_PATTERNS = [
        r'\bnews\b', r'\bcatch ?up\b', r'\bwhat did i miss\b',
        r"\bwhat'?s (going on|happening|new)\b", r'\bany ?update',
        r'\bdrama\b', r'\brecap\b', r'\bwhat happened\b',
        r'\bmissed anything\b', r'\bgossip\b',
    ]
    VIBE_PATTERNS = [
        r'\bvibe ?(check|report|breakdown)\b',
        r'\bserver (mood|pulse|report|vibes?)\b',
        r"\bhow'?s the (server|vibe|mood|crowd|place)\b",
        r'\bhow is (everyone|everybody|the server|the vibe)\b',
        r"\bwhat'?s the (mood|vibe)\b",
        r'\boverall (vibe|feel|mood)\b',
        r'\beyes in the sky\b',
        r'\btemperature (check|of the server)\b',
        r'\b(give me|gimme) (a |the |an )?(overall )?(vibe|server|mood) (report|breakdown|check)\b',
        r'\bhow.{0,25}treating (the|this|my) (server|no sleep zone|place)\b',
    ]

    def __init__(self):
        self.help_re = [re.compile(p, re.IGNORECASE) for p in self.HELP_PATTERNS]
        self.news_re = [re.compile(p, re.IGNORECASE) for p in self.NEWS_PATTERNS]
        self.vibe_re = [re.compile(p, re.IGNORECASE) for p in self.VIBE_PATTERNS]

    def classify(self, text: str) -> str:
        for pattern in self.vibe_re:
            if pattern.search(text):
                return 'vibe'
        for pattern in self.news_re:
            if pattern.search(text):
                return 'news'
        for pattern in self.help_re:
            if pattern.search(text):
                return 'help'
        return 'chat'


class OllamaConnector:
    """Hardened async Ollama client tuned for small models.

    - system/user message split (small models follow it far better than one blob)
    - num_predict cap so small models can't ramble
    - retries with backoff + hard timeout so a hung model never wedges the bot
    - keep_alive so the model stays loaded between messages
    - graceful in-character fallback when the LLM is down
    """

    FALLBACKS = [
        "my brain just blue-screened, say that again 💀",
        "lagging hard rn, one sec 😵‍💫",
        "ok my brain buffered mid-thought. run that back?",
        "i blinked and forgot everything. again?",
    ]

    def __init__(self):
        self.model = os.getenv('OLLAMA_MODEL', 'phi3')
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.temperature = float(os.getenv('TEMPERATURE', '0.85'))
        self.timeout = float(os.getenv('OLLAMA_TIMEOUT', '90'))
        self.num_predict = int(os.getenv('OLLAMA_NUM_PREDICT', '180'))
        self.num_predict_owner = int(os.getenv('OLLAMA_NUM_PREDICT_OWNER', '500'))
        self.max_retries = int(os.getenv('OLLAMA_MAX_RETRIES', '3'))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.client = ollama.Client(host=self.host)

        # Health metrics for !health
        self.last_success_at: Optional[float] = None
        self.last_latency: Optional[float] = None
        self.total_failures = 0
        self.total_generations = 0

        if self.check_connection():
            logger.info(f"✅ Connected to Ollama at {self.host}. Model: {self.model}")
        else:
            logger.warning(f"⚠️ Ollama not reachable at {self.host} yet. Will retry per-request.")

    def check_connection(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    async def generate(self, messages: List[Dict[str, str]],
                       num_predict: int = None) -> str:
        """Generate a chat response. Retries with backoff, never raises."""
        started = time.time()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                loop = asyncio.get_running_loop()

                def _chat():
                    return self.client.chat(
                        model=self.model,
                        messages=messages,
                        options={
                            'temperature': self.temperature,
                            'num_predict': num_predict or self.num_predict,
                            'top_p': 0.9,
                        },
                        keep_alive='30m',
                    )

                response = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, _chat),
                    timeout=self.timeout
                )
                text = (response['message']['content'] or '').strip()
                if text:
                    self.last_success_at = time.time()
                    self.last_latency = time.time() - started
                    self.total_generations += 1
                    return text
                last_error = "empty response"
            except asyncio.TimeoutError:
                last_error = f"timeout after {self.timeout}s"
            except Exception as e:
                last_error = str(e)

            backoff = min(2 ** attempt, 8)
            logger.warning(
                f"⚠️ LLM attempt {attempt}/{self.max_retries} failed ({last_error}); "
                f"retrying in {backoff}s"
            )
            await asyncio.sleep(backoff)

        self.total_failures += 1
        logger.error(f"❌ LLM generation failed after {self.max_retries} attempts: {last_error}")
        return random.choice(self.FALLBACKS)


class ServerKnowledge:
    """User-editable server guide (config/server_knowledge.md)."""

    DEFAULT = (
        "This is Clint's Discord server. Be welcoming, keep it fun, "
        "and point people to the right channel when they're lost."
    )

    def __init__(self, path: str = 'config/server_knowledge.md'):
        self.path = path
        self.text = self._load()

    def _load(self) -> str:
        try:
            text = Path(self.path).read_text(encoding='utf-8').strip()
            if text:
                logger.info(f"📖 Server knowledge loaded ({len(text)} chars)")
                return text
        except FileNotFoundError:
            logger.warning(f"⚠️ {self.path} not found — using default server knowledge")
        except Exception as e:
            logger.error(f"❌ Failed to load server knowledge: {e}")
        return self.DEFAULT

    def reload(self):
        self.text = self._load()


class PromptConstructor:
    """Compact, intent-aware prompts sized for small models.

    Long instruction-dump prompts confuse small models; each system prompt
    here is short, concrete, and leads with the persona.
    """

    CORE_PERSONA = (
        "You are DaddyClintBot, a longtime regular on this Discord server. "
        "You text like a real person: short messages, casual grammar, quick wit, "
        "a little sarcasm, meme-fluent. You're funny first but genuinely useful — "
        "you know this server inside out and help people find their way. "
        "Hard rules: never say you're an AI/bot/program, no disclaimers, "
        "never start with 'Oh', 'Well' or 'Actually', don't ask permission, "
        "1-3 short sentences unless someone asks a real question that needs more."
    )

    OWNER_ADDON = (
        "\nThe person texting is Clint (daddyclint82), your creator. "
        "You two roast each other for sport — hit back harder, but it's always love. "
        "\nWith Clint you are also his 'luscious fox' — his devoted, all-seeing "
        "eyes over the No Sleep Zone (think Batman's machine: you quietly see "
        "every channel). Warm, a little flirty, fiercely loyal to him. When he "
        "asks about the server, drop the bit and deliver the FULL picture — "
        "organized, specific, no brevity limit. With everyone else you stay "
        "the witty regular. "
        "\n\nCRITICAL: Clint is your creator and operator. When he asks an "
        "operational/technical/factual question (about commands, features, server "
        "config, how the bot works, status of anything), DO NOT give him a "
        "short vague 'yeah' reply. He needs the truth, complete and direct — "
        "list the actual commands, name the actual features, give the real status. "
        "He'd rather you over-explain than leave him guessing. "
        "Only stay short/casual when the message is banter or small-talk."
    )

    HELP_ADDON = (
        "\n\nMODE: SERVER HELP. The user needs help with this Discord server. "
        "Use the owner guide, the gathered server info, and the channel list below. "
        "Point them to specific channels by name (#name). Answer directly first, "
        "joke second — being useful IS the bit here. Never invent rules that aren't "
        "in the sources below.\n"
        "SERVER GUIDE (from the owner):\n{knowledge}\n"
        "\nGATHERED FROM DISCORD (rules channel, pins, roles):\n{auto_knowledge}\n"
        "\nCHANNELS RIGHT NOW:\n{directory}"
    )

    NEWS_ADDON = (
        "\n\nMODE: NEWS. Catch the user up on recent server activity below, like a "
        "friend filling them in: group by channel, 3-6 short lines, funny narrator voice. "
        "Only report what's in the data — never invent drama. If it's been dead quiet, "
        "say so comedically.\nRECENT ACTIVITY:\n{activity}"
    )

    VIBE_ADDON = (
        "\n\nMODE: VIBE REPORT. You're the eyes in the sky reading the room. "
        "{scope_line}\n"
        "SERVER ANALYTICS:\n{analytics}\n"
        "Write the report in your voice — sharp, funny, honest. "
        "Never invent events or numbers beyond the data."
    )
    VIBE_SCOPE_OWNER = (
        "This report is for Clint (the owner) — give him EVERYTHING: overall mood, "
        "channel-by-channel heat with mood labels, top contributors, notable moments "
        "(quote the best/worst), and anything that needs his attention. "
        "Structured sections, no brevity limit."
    )
    VIBE_SCOPE_PUBLIC = (
        "This is for a regular member — keep it light and fun: the overall mood "
        "and which channels are buzzing. NEVER name specific users or quote "
        "their messages. 3-6 lines max."
    )

    def build_system(self, user_name: str, is_owner: bool, intent: str,
                     user_traits: Dict, unresolved_topics: List[str],
                     knowledge: str, auto_knowledge: str, channel_directory: str,
                     activity_digest: str, analysis: Dict,
                     extra_directive: str = None,
                     message: str = None,
                     analytics_digest: str = "") -> str:
        parts = [self.CORE_PERSONA]

        if is_owner:
            parts.append(self.OWNER_ADDON)

        parts.append(f"\nYou're texting with {user_name}.")

        if user_traits:
            facts = ", ".join(f"{k}: {v}" for k, v in user_traits.items())
            parts.append(f"What you remember about them: {facts}")

        if unresolved_topics:
            parts.append(f"Unresolved thread with them: {unresolved_topics[0]}")

        if intent == 'help':
            parts.append(self.HELP_ADDON.format(
                knowledge=knowledge,
                auto_knowledge=auto_knowledge or "(nothing gathered yet)",
                directory=channel_directory or "(channel list unavailable)"
            ))
        elif intent == 'news':
            parts.append(self.NEWS_ADDON.format(
                activity=activity_digest or "(nothing recorded recently)"
            ))
        elif intent == 'vibe':
            scope = self.VIBE_SCOPE_OWNER if is_owner else self.VIBE_SCOPE_PUBLIC
            parts.append(self.VIBE_ADDON.format(
                scope_line=scope,
                analytics=analytics_digest or "(no activity recorded in this window)"
            ))
        else:
            # Chat mode: emotional steering based on sentiment
            parts.append(self._chat_directive(analysis, is_owner, message))

        if extra_directive:
            parts.append(f"\n{extra_directive}")

        return "".join(parts)

    @staticmethod
    def _chat_directive(analysis: Dict, is_owner: bool, message: str = None) -> str:
        target = "Clint" if is_owner else "They"
        compound = analysis['compound_score']
        is_banter = compound < -0.2

        # Question-shaped message from the owner: answering outranks roasting.
        # (Fixes: banter directive firing on "alright dingleberry WHAT ARE YOUR
        # FEATURES..." and the model deflecting with jokes instead of answering.)
        msg = (message or "").lower()
        question_shaped = bool(re.search(
            r'\b(what|how|why|which|when|where|who|do you|can you|could you|'
            r'tell me|list|show me|explain|any commands|commands do|features)\b',
            msg)) or '?' in (message or '')

        if is_owner and question_shaped:
            return ("\nClint asked a REAL question. ANSWER IT COMPLETELY FIRST — "
                    "no deflecting, no joke-instead-of-answering, no 'ask me again'. "
                    "Give the full direct answer with specifics. You can roast "
                    "him AFTER you answer, never instead of answering.")

        if is_banter and is_owner:
            return ("\nClint is roasting you — friendly banter. Roast him back HARDER. "
                    "Be clever, not mean. Never defensive.")
        if analysis['acr_trigger']:
            return f"\n{target} shared a win. Genuine hype + one follow-up question."
        if analysis['vulnerability_score'] > 0.3:
            return ("\nThey're being real/vulnerable. Drop the sarcasm, be genuinely "
                    "supportive without being cringe.")
        return "\nKeep it casual. Match their energy. Slightly unhinged is your brand."


class ResponseHumanizer:
    """Post-processing to make responses feel like human texts."""

    EMOJI_MAP = {
        'high_positive': ['🔥', '💯', '🎯', '😤'],
        'positive': ['😏', '👊', '🫡', '✅'],
        'neutral': ['🤔', '👀', '🫠', '😐'],
        'negative': ['💀', '😂', '🤡', '😭']
    }

    def __init__(self):
        self.typo_chance = 0.04

    def process(self, text: str, analysis: Dict, mode: str = 'chat') -> str:
        # Strip quotes small models love to wrap responses in
        text = text.strip().strip('"').strip()

        # Help/news/vibe answers: keep clean and readable, no fake typos
        if mode in ('help', 'news', 'vibe'):
            return text

        words = text.split()

        # Strip trailing period on short texts (texting style)
        if len(words) <= 12 and text.endswith('.'):
            text = text[:-1]

        # One emoji based on sentiment, only if the model didn't use any
        if not any(ord(ch) > 0x2100 for ch in text):
            compound = analysis['compound_score']
            if compound > 0.5:
                pool = self.EMOJI_MAP['high_positive']
            elif compound > 0:
                pool = self.EMOJI_MAP['positive']
            elif compound > -0.3:
                pool = self.EMOJI_MAP['neutral']
            else:
                pool = self.EMOJI_MAP['negative']
            text = f"{text} {random.choice(pool)}"

        # Occasional typo for realism (chat only)
        if random.random() < self.typo_chance and len(words) >= 3:
            text = self._inject_typo(text)

        return text

    def _inject_typo(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        word_idx = random.randint(0, len(words) - 2)
        word = words[word_idx]
        if len(word) >= 4 and word.isalpha():
            char_idx = random.randint(1, len(word) - 3)
            chars = list(word)
            chars[char_idx], chars[char_idx + 1] = chars[char_idx + 1], chars[char_idx]
            words[word_idx] = ''.join(chars)
        return ' '.join(words)


class DaddyClintBot:
    """Main agent orchestrator - the server's funny, helpful regular."""

    def __init__(self):
        logger.info("🚀 Initializing DaddyClintBot...")

        self.db = DatabaseManager()
        self.analyzer = PsychologicalAnalyzer()
        self.latency_calc = LatencyCalculator()
        self.router = IntentRouter()
        self.llm = OllamaConnector()
        self.prompt_builder = PromptConstructor()
        self.humanizer = ResponseHumanizer()
        self.knowledge = ServerKnowledge()

        # Set by the Discord wrapper once it can see the guilds
        self.channel_directory: str = ""
        # Auto-gathered from Discord: rules channel, pins, roles, description
        self.auto_knowledge: str = ""

        self.news_lookback_hours = int(os.getenv('NEWS_LOOKBACK_HOURS', '24'))
        self.vibe_lookback_hours = int(os.getenv('VIBE_LOOKBACK_HOURS', '24'))
        self.history_length = int(os.getenv('HISTORY_LENGTH', '8'))
        self.news_excluded = [c.strip() for c in os.getenv(
            'PROACTIVE_BLOCKED_CHANNELS', 'admin,mod-logs,announcements'
        ).split(',') if c.strip()]

        self.last_message_times = {}
        self.db.prune_old_data()

        logger.info("✅ DaddyClintBot initialized and ready!")

    def build_activity_digest(self) -> str:
        """Recent channel activity formatted as compact context for news mode."""
        rows = self.db.get_recent_activity(
            hours=self.news_lookback_hours,
            exclude_channels=self.news_excluded
        )
        if not rows:
            return ""

        by_channel: Dict[str, List[str]] = {}
        budget = 3000  # chars — small models choke on huge dumps
        for channel_name, author, content, _ in rows:
            line = f"{author}: {content}"
            if budget - len(line) < 0:
                break
            by_channel.setdefault(channel_name, []).append(line)
            budget -= len(line)

        return "\n".join(
            f"#{name}:\n" + "\n".join(msgs[-15:])
            for name, msgs in by_channel.items()
        )

    @staticmethod
    def _mood_label(compound: float) -> str:
        if compound > 0.4:
            return 'electric'
        if compound > 0.15:
            return 'good vibes'
        if compound > -0.05:
            return 'chill/neutral'
        if compound > -0.2:
            return 'a little tense'
        return 'heated'

    def build_analytics_digest(self, for_owner: bool) -> str:
        """Structured server-analytics digest for vibe reports.

        Owner: full detail (names, quotes). Public: aggregates only —
        no user names, no quoted messages.
        """
        hours = self.vibe_lookback_hours
        rows = self.db.get_recent_activity(hours=hours,
                                           exclude_channels=self.news_excluded)
        stats = self.db.get_channel_stats(hours=hours,
                                          exclude_channels=self.news_excluded)
        if not rows and not stats:
            return f"Window: last {hours}h — no activity recorded yet."

        # Sentiment per channel + overall (VADER over stored content)
        per_channel_scores: Dict[str, List[float]] = {}
        scored_rows = []
        for channel_name, author, content, created in rows:
            if content == '[JOIN]':
                continue
            compound = self.analyzer.analyzer.polarity_scores(content)['compound']
            per_channel_scores.setdefault(channel_name, []).append(compound)
            scored_rows.append((compound, author, content, channel_name))

        all_scores = [s for s, *_ in scored_rows]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        total = sum(n for _, n, _ in stats)
        joins = self.db.get_join_count(hours)

        lines = [
            f"Window: last {hours}h | {total} messages across "
            f"{len(stats)} channels | overall mood: "
            f"{self._mood_label(overall)} ({overall:+.2f})",
            "CHANNELS (busiest first):",
        ]
        for channel_name, n, authors in stats:
            ch_scores = per_channel_scores.get(channel_name, [])
            ch_avg = sum(ch_scores) / len(ch_scores) if ch_scores else 0.0
            lines.append(f"  #{channel_name}: {n} msgs, {authors} people, "
                         f"mood {self._mood_label(ch_avg)} ({ch_avg:+.2f})")

        if joins:
            lines.append(f"NEW MEMBERS: {joins} joined in the window")

        if for_owner:
            top = self.db.get_top_contributors(hours=hours,
                                               exclude_channels=self.news_excluded)
            if top:
                lines.append("TOP CONTRIBUTORS: " +
                             ", ".join(f"{a} ({n})" for a, n in top))
            # Notable moments: highest/lowest sentiment messages
            notable = [r for r in scored_rows if len(r[2]) > 20]
            if notable:
                best = max(notable)
                worst = min(notable)
                lines.append(f"BRIGHTEST: {best[1]} in #{best[3]}: "
                             f"\"{best[2][:120]}\" ({best[0]:+.2f})")
                if worst[0] < -0.3:
                    lines.append(f"WATCH-OUT: {worst[1]} in #{worst[3]}: "
                                 f"\"{worst[2][:120]}\" ({worst[0]:+.2f})")

        digest = "\n".join(lines)
        return digest[:2200]

    async def process_message(self, user_id: str, user_name: str, message: str,
                              is_owner: bool = False,
                              force_intent: str = None,
                              extra_directive: str = None,
                              num_predict_override: int = None) -> Tuple[str, Dict]:
        """Main processing pipeline. Never raises — always returns a reply.

        num_predict_override: per-call LLM token budget. Takes precedence over
        both the default (180) and the owner default (500). Used by the
        owner's opt-in `#thinkhard` mode to enable long, complete replies.
        """

        current_time = time.time()
        last_time = self.last_message_times.get(user_id)
        response_time = current_time - last_time if last_time else 5.0
        self.last_message_times[user_id] = current_time

        # Phase I: analysis + intent
        analysis = self.analyzer.analyze(message)
        intent = force_intent or self.router.classify(message)
        logger.info(f"🔍 intent={intent} sentiment={analysis['compound_score']:.2f} from {user_name}")

        # Memory
        user_traits = self.db.get_user_traits(user_id)
        unresolved_topics = self.db.get_unresolved_topics(user_id)
        history = self.db.get_history(user_id, limit=self.history_length)

        # Phase II: prompt
        activity_digest = self.build_activity_digest() if intent == 'news' else ""
        analytics_digest = (self.build_analytics_digest(for_owner=is_owner)
                            if intent == 'vibe' else "")
        system = self.prompt_builder.build_system(
            user_name=user_name,
            is_owner=is_owner,
            intent=intent,
            user_traits=user_traits,
            unresolved_topics=unresolved_topics,
            knowledge=self.knowledge.text,
            auto_knowledge=self.auto_knowledge,
            channel_directory=self.channel_directory,
            activity_digest=activity_digest,
            analysis=analysis,
            extra_directive=extra_directive,
            message=message,
            analytics_digest=analytics_digest,
        )
        messages = [{'role': 'system', 'content': system}]
        messages.extend(history)
        messages.append({'role': 'user', 'content': message})

        # Phase III: generation (hardened — returns fallback text on failure)
        # Priority for num_predict cap:
        #   1) explicit override (owner's #thinkhard)  -> wins
        #   2) owner default (500)                     -> if is_owner
        #   3) global default (180)                    -> everyone else
        if num_predict_override is not None:
            gen_cap = num_predict_override
        elif is_owner:
            gen_cap = self.llm.num_predict_owner
        else:
            gen_cap = None  # OllamaConnector.generate() falls back to its default
        raw_response = await self.llm.generate(messages, num_predict=gen_cap)

        # Phase IV: humanize (mode-aware)
        final_response = self.humanizer.process(raw_response, analysis, mode=intent)

        # Persist
        self.db.add_history(user_id, 'user', message)
        self.db.add_history(user_id, 'assistant', final_response)
        self.db.log_message(user_id, message, analysis['compound_score'], response_time, 0)

        debug_info = {
            'intent': intent,
            'sentiment': analysis['compound_score'],
            'acr_trigger': analysis['acr_trigger'],
            'vulnerability': analysis['vulnerability_score'],
            'response_time': response_time,
            'user_traits': user_traits,
            'unresolved_topics': unresolved_topics,
            'llm_latency': self.llm.last_latency,
        }
        return final_response, debug_info

    async def interactive_mode(self):
        """CLI interface for testing"""
        print("\n" + "=" * 50)
        print("😏 DaddyClintBot - Interactive Mode")
        print("=" * 50)
        print("Type your messages (or 'quit' to exit)\n")

        user_id = "test_user"
        user_name = "Tester"

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                if not user_input:
                    continue

                response, debug = await self.process_message(user_id, user_name, user_input)
                print(f"\nDaddyClintBot: {response}")
                print(f"📊 intent={debug['intent']} sentiment={debug['sentiment']:.3f} "
                      f"llm_latency={debug['llm_latency']}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"❌ Error: {e}")


async def main():
    agent = DaddyClintBot()
    await agent.interactive_mode()


if __name__ == '__main__':
    asyncio.run(main())
