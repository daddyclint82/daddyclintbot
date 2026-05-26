import asyncio
import logging
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

import asyncio
import concurrent.futures
from functools import partial

import ollama

# Load environment variables
load_dotenv()

# Configure logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daddyclintbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DaddyClintBot')


class DatabaseManager:
    """SQLite database manager for state and memory"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv('DB_PATH', 'data/daddyclintbot.db')
        Path(self.db_path).parent.mkdir(exist_ok=True)
        self.init_db()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            # Conversation state tracking (Zeigarnik Effect)
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
            
            # User traits and shared secrets
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
            
            # Message history
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
            
            conn.commit()
    
    def get_unresolved_topics(self, user_id: str, limit: int = 1) -> List[str]:
        """Get unresolved topics (Zeigarnik Effect)"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT topic FROM ConversationState 
                WHERE user_id = ? AND is_unresolved = 1
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
            return [row[0] for row in cursor.fetchall()]
    
    def set_topic_resolved(self, user_id: str, topic: str):
        """Mark a topic as resolved"""
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE ConversationState 
                SET is_unresolved = 0, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND topic = ?
            ''', (user_id, topic))
            conn.commit()
    
    def add_topic(self, user_id: str, topic: str, is_unresolved: bool = True):
        """Add a new conversation topic"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO ConversationState (user_id, topic, is_unresolved, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, topic, is_unresolved))
            conn.commit()
    
    def get_user_traits(self, user_id: str, limit: int = 3) -> Dict[str, str]:
        """Get user facts for charismatic recall"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT trait_key, trait_value FROM UserTraits
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            return dict(cursor.fetchall())
    
    def add_user_trait(self, user_id: str, key: str, value: str):
        """Store a user trait/fact"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO UserTraits (user_id, trait_key, trait_value)
                VALUES (?, ?, ?)
            ''', (user_id, key, value))
            conn.commit()
    
    def get_last_message_time(self, user_id: str) -> Optional[datetime]:
        """Get timestamp of last message from user"""
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
        """Log message for analytics"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO Messages (user_id, content, sentiment_score, response_time, bot_delay)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, content, sentiment_score, response_time, bot_delay))
            conn.commit()


class PsychologicalAnalyzer:
    """Lightweight NLP analysis using VADER sentiment"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment and extract psychological flags"""
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Active Constructive Responding trigger
        acr_trigger = compound > 0.7
        
        # Detect self-disclosure (vulnerability)
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
    """Calculate Poisson-based response delays"""
    
    def __init__(self):
        self.min_delay = float(os.getenv('MIN_DELAY', '1.0'))
        self.max_delay = float(os.getenv('MAX_DELAY', '15.0'))
    
    def calculate_delay(self, user_response_time: float) -> float:
        """Calculate bot delay using Poisson distribution"""
        # Poisson distribution for natural-feeling delays
        lambda_param = user_response_time * 1.1  # Slightly longer than user
        delay = np.random.poisson(lambda_param)
        
        # Clamp to reasonable bounds
        return min(max(delay, self.min_delay), self.max_delay)


class OllamaConnector:
    """Modular LLM interface for Ollama - async wrapper"""
    
    def __init__(self):
        self.model = os.getenv('OLLAMA_MODEL', 'phi3')
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.temperature = float(os.getenv('TEMPERATURE', '0.85'))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Test connection (non-blocking, graceful fallback)
        try:
            client = ollama.Client(host=self.host)
            client.list()
            logger.info(f"✅ Connected to Ollama at {self.host}. Model: {self.model}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama connection check failed: {e}. Will retry on first generation.")
    
    async def generate(self, prompt: str, context: Dict = None) -> str:
        """Generate response from Ollama model - runs in thread pool"""
        try:
            loop = asyncio.get_event_loop()
            
            def _chat():
                client = ollama.Client(host=self.host)
                return client.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': self.temperature}
                )
            
            response = await loop.run_in_executor(self.executor, _chat)
            return response['message']['content']
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "My brain lagged, say that again? 💀"


class PromptConstructor:
    """Builds dynamic prompts with psychological triggers"""
    
    # Static knowledge about the owner - this never changes
    OWNER_CONTEXT = """CRITICAL CONTEXT:
Your name is DaddyClintBot. You are owned by a man named Clint (username: daddyclint82).
Clint is your creator and boss. You talk to him like a close friend who roasts him but secretly respects him.
You are NOT a generic AI assistant - you are Clint's personal bot.
When Clint insults you, it's banter. Roast him back. When he's being sarcastic, match his energy.
Never explain that you're an AI. Never say "as an AI." Never break character.
Your personality: witty, sarcastic, confident, occasionally absurd, but always loyal to Clint."""
    
    @staticmethod
    def construct(user_message: str, user_traits: Dict, unresolved_topics: List[str],
                  analysis: Dict) -> str:
        """Construct the prompt with context injection"""
        
        # Build context section
        context_parts = [PromptConstructor.OWNER_CONTEXT]
        
        if user_traits:
            facts = ", ".join([f"{k}: {v}" for k, v in user_traits.items()])
            context_parts.append(f"Things you remember about Clint: {facts}")
        
        if unresolved_topics:
            context_parts.append(f"Previous unresolved topic you were discussing: {unresolved_topics[0]}")
        
        # Detect if this is Clint roasting/bantering
        is_banter = analysis['compound_score'] < -0.2 and ('you' in user_message.lower() or 'bot' in user_message.lower() or 'brain' in user_message.lower())
        
        # Determine directive
        if is_banter:
            directive = "Clint is roasting you. This is friendly banter. Roast him back HARDER. Be clever, not mean. Make him laugh. Under no circumstances should you be defensive or explain yourself."
        elif analysis['acr_trigger']:
            directive = "Clint is sharing something positive. Respond with genuine enthusiasm. Ask a follow-up that shows you're actually interested."
        elif analysis['vulnerability_score'] > 0.3:
            directive = "Clint is being vulnerable. Drop the sarcasm and be genuinely supportive. Acknowledge his feelings without being cringe."
        else:
            directive = "Keep it casual. Match Clint's energy. If he's being weird, be weirder. If he's being normal, be slightly unhinged."
        
        # Construct full prompt
        prompt = f"""{chr(10).join(context_parts)}

Clint just said: "{user_message}"

{directive}

ABSOLUTE RULES - NEVER BREAK THESE:
- 1-3 sentences max. Brevity is wit.
- NEVER explain that you're an AI, bot, program, or machine
- NEVER say "as an AI", "I don't have feelings", "I'm just a bot", "I'm just a program"
- NEVER ask permission or be overly polite
- NEVER start with "Oh" or "Well" or "Actually"
- NEVER use disclaimers or explanations
- If Clint insults you, fire back with better insults
- If Clint asks a personal question about him, admit you don't know by making something absurd up
- Sound like a real person texting their friend
- Emojis are fine but don't overdo it"""

        return prompt


class ResponseHumanizer:
    """Post-processing to make responses feel more human"""
    
    # Emoji mapping based on sentiment
    EMOJI_MAP = {
        'high_positive': ['🔥', '💯', '🎯', '😤'],
        'positive': ['😏', '👊', '🫡', '✅'],
        'neutral': ['🤔', '👀', '🫠', '😐'],
        'negative': ['💀', '😂', '🤡', '😭']
    }
    
    def __init__(self):
        self.typo_chance = 0.05  # 5% typo injection
    
    def process(self, text: str, analysis: Dict) -> str:
        """Apply humanization rules"""
        
        # Rule 1: Strip periods on short sentences
        words = text.split()
        if len(words) <= 12 and text.endswith('.'):
            text = text[:-1]
        
        # Rule 2: Emoji injection based on sentiment
        compound = analysis['compound_score']
        if compound > 0.5:
            emoji_pool = self.EMOJI_MAP['high_positive']
        elif compound > 0:
            emoji_pool = self.EMOJI_MAP['positive']
        elif compound > -0.3:
            emoji_pool = self.EMOJI_MAP['neutral']
        else:
            emoji_pool = self.EMOJI_MAP['negative']
        
        emoji = random.choice(emoji_pool)
        text = f"{text} {emoji}"
        
        # Rule 3: Typo injection (5% chance)
        if random.random() < self.typo_chance and len(words) >= 3:
            text = self._inject_typo(text)
        
        return text
    
    def _inject_typo(self, text: str) -> str:
        """Occasional typo to mimic mobile typing"""
        # Simple typo: swap adjacent letters in a word
        words = text.split()
        if len(words) < 2:
            return text
        
        # Pick a random word (not the last one, and not punctuation)
        word_idx = random.randint(0, len(words) - 2)
        word = words[word_idx]
        
        if len(word) >= 4:
            # Swap two adjacent letters in the middle
            char_idx = random.randint(1, len(word) - 3)
            chars = list(word)
            chars[char_idx], chars[char_idx + 1] = chars[char_idx + 1], chars[char_idx]
            words[word_idx] = ''.join(chars)
        
        return ' '.join(words)


class DaddyClintBot:
    """Main agent orchestrator - The Charming Witty Asshole"""
    
    def __init__(self):
        logger.info("🚀 Initializing DaddyClintBot...")
        
        self.db = DatabaseManager()
        self.analyzer = PsychologicalAnalyzer()
        self.latency_calc = LatencyCalculator()
        self.llm = OllamaConnector()
        self.prompt_builder = PromptConstructor()
        self.humanizer = ResponseHumanizer()
        
        # Track last message time per user
        self.last_message_times = {}
        
        logger.info("✅ DaddyClintBot initialized and ready to charm!")
    
    async def process_message(self, user_id: str, message: str) -> Tuple[str, Dict]:
        """Main processing pipeline"""
        
        # Calculate response time
        current_time = time.time()
        last_time = self.last_message_times.get(user_id)
        response_time = current_time - last_time if last_time else 5.0
        self.last_message_times[user_id] = current_time
        
        # Phase I: Lightweight Analysis
        logger.info("🔍 Phase I: Analyzing sentiment...")
        analysis = self.analyzer.analyze(message)
        
        # Calculate delay
        bot_delay = self.latency_calc.calculate_delay(response_time)
        
        # Retrieve memory
        user_traits = self.db.get_user_traits(user_id)
        unresolved_topics = self.db.get_unresolved_topics(user_id)
        
        logger.info(f"🧠 Retrieved {len(user_traits)} traits, {len(unresolved_topics)} unresolved topics")
        
        # Phase II: Prompt Construction
        logger.info("📝 Phase II: Constructing prompt...")
        prompt = self.prompt_builder.construct(
            message, user_traits, unresolved_topics, analysis
        )
        
        # Phase III: LLM Generation
        logger.info("🤖 Phase III: Generating response...")
        raw_response = await self.llm.generate(prompt)
        
        # Phase IV: Humanization
        logger.info("✨ Phase IV: Humanizing...")
        final_response = self.humanizer.process(raw_response, analysis)
        
        # Log to database
        self.db.log_message(user_id, message, analysis['compound_score'], 
                           response_time, bot_delay)
        
        # Simulate the calculated delay
        logger.info(f"⏱️  Waiting {bot_delay:.1f}s before responding...")
        await asyncio.sleep(bot_delay)
        
        # Prepare debug info
        debug_info = {
            'sentiment': analysis['compound_score'],
            'acr_trigger': analysis['acr_trigger'],
            'vulnerability': analysis['vulnerability_score'],
            'response_time': response_time,
            'bot_delay': bot_delay,
            'user_traits': user_traits,
            'unresolved_topics': unresolved_topics
        }
        
        return final_response, debug_info
    
    async def interactive_mode(self):
        """CLI interface for testing"""
        print("\n" + "="*50)
        print("😏 DaddyClintBot - Interactive Mode")
        print("="*50)
        print("Type your messages (or 'quit' to exit)\n")
        
        user_id = "test_user"
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🔄 Processing...")
                response, debug = await self.process_message(user_id, user_input)
                
                print(f"\n{'='*50}")
                print(f"DaddyClintBot: {response}")
                print(f"{'='*50}")
                print(f"📊 Debug Info:")
                print(f"   Sentiment: {debug['sentiment']:.3f}")
                print(f"   ACR Trigger: {'✅' if debug['acr_trigger'] else '❌'}")
                print(f"   Vulnerability: {debug['vulnerability']:.3f}")
                print(f"   Response Time: {debug['response_time']:.1f}s")
                print(f"   Bot Delay: {debug['bot_delay']:.1f}s")
                print(f"   Traits: {debug['user_traits']}")
                print(f"   Unresolved: {debug['unresolved_topics']}")
                print(f"{'='*50}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"❌ Error: {e}")


async def main():
    """Entry point"""
    agent = DaddyClintBot()
    await agent.interactive_mode()


if __name__ == '__main__':
    asyncio.run(main())