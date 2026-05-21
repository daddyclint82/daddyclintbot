import asyncio
import logging
import os
import sys
from pathlib import Path

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from agent import DaddyClintBot
import time

# Load environment variables
load_dotenv()

# Configure logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DiscordBot')


class DaddyClintDiscordBot(commands.Bot):
    """Discord bot wrapper for DaddyClintBot psychological engine"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.presences = True
        
        super().__init__(
            command_prefix=os.getenv('BOT_PREFIX', '!'),
            intents=intents,
            case_insensitive=True
        )
        
        # Initialize psychological engine
        self.psych_bot = None
        
        # Track message timestamps per user
        self.last_message_times = {}
        
        # === PROACTIVE ENGAGEMENT CONFIGURATION ===
        # Enable/disable proactive engagement entirely
        self.proactive_enabled = os.getenv('PROACTIVE_ENABLED', 'true').lower() == 'true'
        
        # Confidence threshold (0.0-1.0): higher = more selective, lower = more chatty
        # Recommended: 0.6-0.8 for production, 0.4-0.5 for testing
        # INCREASED to 0.75 to reduce unwanted proactive engagement
        self.proactive_threshold = float(os.getenv('PROACTIVE_THRESHOLD', '0.75'))
        
        # Cooldown settings (in seconds)
        self.channel_cooldown_seconds = int(os.getenv('PROACTIVE_CHANNEL_COOLDOWN', '300'))  # 5 min
        self.user_cooldown_seconds = int(os.getenv('PROACTIVE_USER_COOLDOWN', '60'))  # 1 min
        
        # Minimum message length to consider
        self.min_message_length = int(os.getenv('PROACTIVE_MIN_LENGTH', '15'))
        
        # Allowed channel names (empty = all channels)
        # Example: "general,gaming,fitness"
        allowed_channels = os.getenv('PROACTIVE_CHANNELS', '')
        self.allowed_channels = [c.strip() for c in allowed_channels.split(',') if c.strip()]
        
        # Blocked channel names (always excluded)
        blocked_channels = os.getenv('PROACTIVE_BLOCKED_CHANNELS', 'admin,mod-logs,announcements')
        self.blocked_channels = [c.strip() for c in blocked_channels.split(',') if c.strip()]
        
        # Topic keywords for engagement scoring boost
        # Gaming, tech, fitness domains
        self.topic_keywords = {
            'gaming': ['game', 'gaming', 'play', 'played', 'valorant', 'cod', 'fps', 'rpg', 'elden', 'dark souls', 'rage quit', 'win', 'loss', 'rank', 'competitive'],
            'tech': ['code', 'coding', 'program', 'bug', 'debug', 'deploy', 'server', 'api', 'javascript', 'python', 'rust', 'error', 'compile', 'git', 'github'],
            'fitness': ['gym', 'workout', 'lift', 'bench', 'squat', 'run', 'cardio', 'protein', 'gain', 'cut', 'bulk', 'prs', 'personal record', 'leg day', 'rest day'],
            'general': ['tired', 'exhausted', 'stressed', 'celebrate', 'win', 'achievement', 'grind', 'hustle', 'burnout', 'motivation']
        }
        
        # Tracking for cooldowns
        self.last_bot_activity = {}  # channel_id -> timestamp
        self.last_user_engagement = {}  # user_id -> timestamp
        
        # Dry-run mode: log but don't send (for testing thresholds)
        self.proactive_dry_run = os.getenv('PROACTIVE_DRY_RUN', 'false').lower() == 'true'
        
        # Bot status
        self.activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="for DMs | !help"
        )
    
    async def setup_hook(self):
        """Called when bot is starting up"""
        logger.info("🚀 Initializing DaddyClintBot psychological engine...")
        try:
            self.psych_bot = DaddyClintBot()
            logger.info("✅ Psychological engine loaded!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize psychological engine: {e}")
            raise
        
        # Start background tasks
        self.status_update.start()
    
    async def on_ready(self):
        """Called when bot is fully connected"""
        logger.info(f'✅ {self.user} has connected to Discord!')
        logger.info(f'🌐 Bot is in {len(self.guilds)} guilds')
        
        await self.change_presence(activity=self.activity)
    
    async def on_message(self, message):
        """Handle incoming messages"""
        # Ignore bot's own messages
        if message.author == self.user:
            return
        
        # Process commands first
        await self.process_commands(message)
        
        # Handle DMs with psychological engine
        if isinstance(message.channel, discord.DMChannel):
            await self._handle_dm(message)
        
        # Handle mentions in guild channels
        elif self.user in message.mentions:
            await self._handle_mention(message)
        
        # === NEW: Proactive channel monitoring ===
        elif isinstance(message.channel, discord.TextChannel) and self.proactive_enabled:
            await self._evaluate_proactive_engagement(message)
    
    async def _handle_dm(self, message: discord.Message):
        """Handle direct messages with psychological engagement"""
        user_id = str(message.author.id)
        content = message.content
        
        logger.info(f"📨 DM from {message.author}: {content[:50]}...")
        
        # NEW: Skip empty or link-only messages
        if not content or content.strip().startswith('http'):
            return
        
        # Start typing indicator as a background task
        typing_task = asyncio.create_task(self._keep_typing(message.channel))
        
        try:
            # Process through psychological engine (without artificial delay for Discord)
            response, debug = await self._process_discord_message(user_id, content)
            
            logger.info(f"🤖 Generated response: {response[:50]}...")
            logger.debug(f"Debug info: {debug}")
            
            # Cancel typing
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            
            # Send response
            await message.reply(response)
            
        except Exception as e:
            logger.error(f"❌ Error processing DM: {e}", exc_info=True)
            typing_task.cancel()
            await message.reply("Oops, my brain glitched. Try that again? 💀")
    
    async def _handle_mention(self, message: discord.Message):
        """Handle mentions in guild channels"""
        # Clean the message content (remove bot mention)
        content = message.content.replace(f'<@{self.user.id}>', '').replace(f'<@!{self.user.id}>', '').strip()
        
        if not content:
            await message.reply("What's up? 👀")
            return
        
        # NEW: Skip if it's just a link after removing mention
        if content.startswith('http') or content.startswith('https'):
            return
        
        user_id = str(message.author.id)
        
        logger.info(f"📣 Mention from {message.author}: {content[:50]}...")
        
        # Start typing indicator as a background task
        typing_task = asyncio.create_task(self._keep_typing(message.channel))
        
        try:
            response, debug = await self._process_discord_message(user_id, content)
            
            # Cancel typing
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            
            await message.reply(response)
        except Exception as e:
            logger.error(f"❌ Error processing mention: {e}", exc_info=True)
            typing_task.cancel()
            await message.reply("Got distracted. What were we talking about? 🤔")
    
    @commands.command(name='status')
    async def status(self, ctx):
        """Show bot status"""
        embed = discord.Embed(
            title="😏 DaddyClintBot Status",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Psych Engine",
            value="✅ Online" if self.psych_bot else "❌ Offline",
            inline=True
        )
        embed.add_field(
            name="Latency",
            value=f"{round(self.latency * 1000)}ms",
            inline=True
        )
        embed.add_field(
            name="Servers",
            value=len(self.guilds),
            inline=True
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='persona')
    async def persona(self, ctx):
        """Show bot persona info"""
        embed = discord.Embed(
            title="😏 Who is DaddyClintBot?",
            description="The Charming Witty Asshole",
            color=discord.Color.purple()
        )
        
        embed.add_field(
            name="Vibe",
            value="Playful, sarcastic, but genuinely caring when it counts",
            inline=False
        )
        embed.add_field(
            name="Features",
            value="• Psychological engagement\n• Memory of your facts\n• Natural conversation flow\n• Active Constructive Responding",
            inline=False
        )
        embed.add_field(
            name="How to chat",
            value="DM me or @mention me in a server!",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    async def _evaluate_proactive_engagement(self, message: discord.Message):
        """
        Analyze guild messages and decide whether to engage proactively.
        Uses configurable thresholds and cooldowns to avoid spam.
        """
        try:
            # --- STEP 1: Quick Filters (cheap checks first) ---
            
            # Skip if message is too short
            if len(message.content) < self.min_message_length:
                return
            
            # NEW: Skip if message is just a link or command
            content_lower = message.content.lower().strip()
            if content_lower.startswith('http') or content_lower.startswith('!') or content_lower.startswith('?'):
                return
            
            # NEW: Skip if message is just an emoji or very short reaction
            if len(message.content) < 20 and not any(c.isalpha() for c in message.content):
                return
            
            channel_id = str(message.channel.id)
            user_id = str(message.author.id)
            current_time = time.time()
            
            # Skip if channel is blocked
            if message.channel.name in self.blocked_channels:
                return
            
            # Skip if allowlist is defined and channel not in it
            if self.allowed_channels and message.channel.name not in self.allowed_channels:
                return
            
            # Skip if bot was recently active in this channel (cooldown)
            last_activity = self.last_bot_activity.get(channel_id, 0)
            if current_time - last_activity < self.channel_cooldown_seconds:
                return
            
            # Skip if we recently engaged with this user (prevent bot from being clingy)
            last_user_eng = self.last_user_engagement.get(user_id, 0)
            if current_time - last_user_eng < self.user_cooldown_seconds:
                return
            
            # NEW: Skip if user is the bot owner (daddyclint82) - don't proactively engage with owner
            if user_id == "1496169097942274208":
                return
            
            # --- STEP 2: Content Analysis ---
            logger.debug(f"🔍 Analyzing proactive opportunity from {message.author}: {message.content[:50]}...")
            
            analysis = self.psych_bot.analyzer.analyze(message.content)
            
            # --- STEP 3: Calculate Engagement Score ---
            engagement_score = self._calculate_engagement_score(message, analysis)
            
            # Always log proactive scores at INFO for visibility
            logger.info(
                f"📊 Proactive score: {engagement_score:.2f} (threshold: {self.proactive_threshold}) | "
                f"Sentiment: {analysis['compound_score']:.2f} | "
                f"ACR: {analysis['acr_trigger']} | "
                f"Vuln: {analysis['vulnerability_score']:.2f} | "
                f"Content: {message.content[:40]}..."
            )
            
            # --- STEP 4: Threshold Check ---
            if engagement_score < self.proactive_threshold:
                return
            
            # --- STEP 5: Dry Run Mode (log but don't send) ---
            if self.proactive_dry_run:
                logger.info(f"🧪 [DRY RUN] Would respond to: {message.content[:50]}... (score: {engagement_score:.2f})")
                return
            
            # --- STEP 6: Generate & Send Response ---
            logger.info(f"🎯 Proactive engagement triggered (score: {engagement_score:.2f}) for {message.author}")
            
            # Update tracking timestamps
            self.last_bot_activity[channel_id] = current_time
            self.last_user_engagement[user_id] = current_time
            
            # Generate response using existing pipeline
            typing_task = asyncio.create_task(self._keep_typing(message.channel))
            
            try:
                # Build prompt for proactive response
                user_traits = self.psych_bot.db.get_user_traits(user_id)
                unresolved_topics = self.psych_bot.db.get_unresolved_topics(user_id)
                
                # Modify system prompt to indicate this is proactive (not a reply)
                prompt = self._build_proactive_prompt(
                    message.content, message.author.display_name, 
                    user_traits, unresolved_topics, analysis
                )
                
                raw_response = await self.psych_bot.llm.generate(prompt)
                final_response = self.psych_bot.humanizer.process(raw_response, analysis)
                
                # Cancel typing
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
                
                # Reply to the message
                await message.reply(final_response)
                logger.info(f"✅ Proactive response sent to {message.author}")
                
            except Exception as e:
                logger.error(f"❌ Error in proactive response: {e}", exc_info=True)
                typing_task.cancel()
        
        except Exception as e:
            logger.error(f"❌ Error in proactive engagement evaluation: {e}", exc_info=True)
    
    def _calculate_engagement_score(self, message: discord.Message, analysis: dict) -> float:
        """
        Calculate engagement opportunity score (0.0 - 1.0).
        Higher = more likely to engage.
        """
        score = 0.0
        content_lower = message.content.lower()
        
        # Base sentiment signals
        sentiment = analysis.get('compound_score', 0)
        
        # Strong negative sentiment (frustration/venting) - high engagement opportunity
        if sentiment < -0.5:
            score += 0.3
        # Strong positive sentiment (celebration) - worth amplifying
        elif sentiment > 0.5:
            score += 0.25
        
        # Vulnerability detected - user might need support
        vuln_score = analysis.get('vulnerability_score', 0)
        if vuln_score > 0.6:
            score += 0.25
        
        # ACR trigger (celebration moment)
        if analysis.get('acr_trigger', False):
            score += 0.2
        
        # Topic keyword matching
        topic_matches = 0
        matched_domains = []
        for domain, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    topic_matches += 1
                    if domain not in matched_domains:
                        matched_domains.append(domain)
        
        # Boost score for relevant topic matches (diminishing returns after 3)
        topic_boost = min(topic_matches * 0.08, 0.25)
        score += topic_boost
        
        # Time-based signals (optional - could detect late night coding/grinding)
        hour = message.created_at.hour if message.created_at else 12
        if hour >= 0 and hour <= 3:  # Late night activity
            score += 0.05  # Small boost - "still up?" energy
        
        # Question detection (user seeking engagement)
        if '?' in message.content:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _build_proactive_prompt(self, content: str, author_name: str, 
                                 user_traits: list, unresolved_topics: list, 
                                 analysis: dict) -> str:
        """
        Build a prompt for proactive engagement.
        Emphasizes that this is an unsolicited but natural interjection.
        """
        sentiment_desc = "positive" if analysis['compound_score'] > 0 else "negative" if analysis['compound_score'] < 0 else "neutral"
        
        base_prompt = f"""You are DaddyClintBot, a Discord user with a personality that's playful, sarcastic, but genuinely caring. You noticed a message in the chat and decided to jump in naturally.

Context:
- User '{author_name}' just said: "{content}"
- Sentiment: {sentiment_desc}
- Vulnerability detected: {"Yes" if analysis['vulnerability_score'] > 0.5 else "No"}

Guidelines for this proactive response:
1. Reference their message specifically - show you actually read it
2. Match their energy but add your personality (witty, slightly sarcastic, caring underneath)
3. Keep it conversational - this isn't a formal reply, it's a friend jumping into the chat
4. If they seem frustrated, roast them gently but validate the struggle
5. If they're celebrating, amplify it with genuine hype
6. If they're vulnerable, be supportive without being cringe
7. 1-3 sentences max - don't dominate the conversation
8. Never start with "As an AI" or explain yourself

Respond as DaddyClintBot:"""
        
        return base_prompt
    
    @commands.command(name='proactive')
    async def proactive_status(self, ctx):
        """Show proactive engagement status and stats"""
        embed = discord.Embed(
            title="🎯 Proactive Engagement Status",
            color=discord.Color.green() if self.proactive_enabled else discord.Color.red()
        )
        
        embed.add_field(
            name="Status",
            value="✅ Enabled" if self.proactive_enabled else "❌ Disabled",
            inline=True
        )
        embed.add_field(
            name="Threshold",
            value=f"{self.proactive_threshold:.2f}",
            inline=True
        )
        embed.add_field(
            name="Dry Run",
            value="✅ Yes (logging only)" if self.proactive_dry_run else "❌ No (live replies)",
            inline=True
        )
        embed.add_field(
            name="Channel Cooldown",
            value=f"{self.channel_cooldown_seconds}s",
            inline=True
        )
        embed.add_field(
            name="User Cooldown",
            value=f"{self.user_cooldown_seconds}s",
            inline=True
        )
        embed.add_field(
            name="Min Length",
            value=f"{self.min_message_length} chars",
            inline=True
        )
        
        if self.allowed_channels:
            embed.add_field(
                name="Allowed Channels",
                value=", ".join(self.allowed_channels),
                inline=False
            )
        
        embed.add_field(
            name="Blocked Channels",
            value=", ".join(self.blocked_channels),
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    async def _keep_typing(self, channel):
        """Keep typing indicator active"""
        while True:
            try:
                await channel.trigger_typing()
                await asyncio.sleep(5)  # Discord typing lasts ~10 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                break
    
    async def _process_discord_message(self, user_id: str, content: str):
        """Process message without artificial delay for Discord"""
        # Calculate response time
        current_time = time.time()
        last_time = self.psych_bot.last_message_times.get(user_id)
        response_time = current_time - last_time if last_time else 5.0
        self.psych_bot.last_message_times[user_id] = current_time
        
        # Phase I: Lightweight Analysis
        logger.info("🔍 Phase I: Analyzing sentiment...")
        analysis = self.psych_bot.analyzer.analyze(content)
        
        # Retrieve memory
        user_traits = self.psych_bot.db.get_user_traits(user_id)
        unresolved_topics = self.psych_bot.db.get_unresolved_topics(user_id)
        
        logger.info(f"🧠 Retrieved {len(user_traits)} traits, {len(unresolved_topics)} unresolved topics")
        
        # Phase II: Prompt Construction
        logger.info("📝 Phase II: Constructing prompt...")
        prompt = self.psych_bot.prompt_builder.construct(
            content, user_traits, unresolved_topics, analysis
        )
        
        # Phase III: LLM Generation
        logger.info("🤖 Phase III: Generating response...")
        raw_response = await self.psych_bot.llm.generate(prompt)
        
        # Phase IV: Humanization
        logger.info("✨ Phase IV: Humanizing...")
        final_response = self.psych_bot.humanizer.process(raw_response, analysis)
        
        # Log to database
        self.psych_bot.db.log_message(user_id, content, analysis['compound_score'], 
                           response_time, 0)  # No artificial delay for Discord
        
        debug_info = {
            'sentiment': analysis['compound_score'],
            'acr_trigger': analysis['acr_trigger'],
            'vulnerability': analysis['vulnerability_score'],
            'response_time': response_time,
            'user_traits': user_traits,
            'unresolved_topics': unresolved_topics
        }
        
        return final_response, debug_info
    
    @tasks.loop(minutes=5)
    async def status_update(self):
        """Update bot status periodically"""
        if self.psych_bot:
            await self.change_presence(activity=self.activity)
    
    @status_update.before_loop
    async def before_status_update(self):
        await self.wait_until_ready()


async def main():
    """Main entry point"""
    token = os.getenv('DISCORD_TOKEN')
    
    if not token:
        logger.error("❌ DISCORD_TOKEN not found in environment!")
        logger.error("Please set it in your .env file")
        sys.exit(1)
    
    bot = DaddyClintDiscordBot()
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
        await bot.close()
    except Exception as e:
        logger.error(f"❌ Bot error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
