"""DaddyClintBot - Discord client.

Resilience model:
    - discord.py auto-reconnects gateway drops; this file handles the fatal cases
      by recreating the bot instance with backoff (a closed Client can't be reused)
    - engine.process_message never raises (LLM failures -> in-character fallback)
    - every handler is wrapped so one bad message can never kill the loop
    - background loops: presence refresh, channel-directory refresh, DB prune,
      Ollama health watchdog
"""

import asyncio
import logging
import logging.handlers
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from agent import DaddyClintBot
from snapshot import run_snapshot

# Load environment variables
load_dotenv()

# Configure logging with rotation
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/discord_bot.log', maxBytes=5_000_000, backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DiscordBot')


class DaddyClintDiscordBot(commands.Bot):
    """Discord bot wrapper for the DaddyClintBot engine"""

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

        self.engine: DaddyClintBot = None
        self.start_time = time.time()
        self.owner_id = os.getenv('OWNER_ID', '1496169097942274208')

        # === OWNER #thinkhard MODE ===
        # Per-user override of LLM num_predict (token budget).
        # Owner-only opt-in. Format: '#thinkhard' = 1000 tokens,
        # '#thinkhard 1500' = custom, '#thinkhard off' = reset to default.
        # State is per-user_id, lives in this dict, resets on bot restart.
        self.owner_token_overrides: Dict[str, int] = {}
        self.DEFAULT_THINKHARD_BUDGET = 1000

        # === PROACTIVE ENGAGEMENT CONFIGURATION ===
        self.proactive_enabled = os.getenv('PROACTIVE_ENABLED', 'true').lower() == 'true'
        self.proactive_threshold = float(os.getenv('PROACTIVE_THRESHOLD', '0.75'))
        self.channel_cooldown_seconds = int(os.getenv('PROACTIVE_CHANNEL_COOLDOWN', '300'))
        self.user_cooldown_seconds = int(os.getenv('PROACTIVE_USER_COOLDOWN', '60'))
        self.min_message_length = int(os.getenv('PROACTIVE_MIN_LENGTH', '15'))

        allowed_channels = os.getenv('PROACTIVE_CHANNELS', '')
        self.allowed_channels = [c.strip() for c in allowed_channels.split(',') if c.strip()]

        blocked_channels = os.getenv('PROACTIVE_BLOCKED_CHANNELS', 'admin,mod-logs,announcements')
        self.blocked_channels = [c.strip() for c in blocked_channels.split(',') if c.strip()]

        self.topic_keywords = {
            'gaming': ['game', 'gaming', 'play', 'played', 'valorant', 'cod', 'fps', 'rpg',
                       'elden', 'dark souls', 'rage quit', 'win', 'loss', 'rank', 'competitive'],
            'tech': ['code', 'coding', 'program', 'bug', 'debug', 'deploy', 'server', 'api',
                     'javascript', 'python', 'rust', 'error', 'compile', 'git', 'github'],
            'fitness': ['gym', 'workout', 'lift', 'bench', 'squat', 'run', 'cardio', 'protein',
                        'gain', 'cut', 'bulk', 'prs', 'personal record', 'leg day', 'rest day'],
            'general': ['tired', 'exhausted', 'stressed', 'celebrate', 'win', 'achievement',
                        'grind', 'hustle', 'burnout', 'motivation']
        }

        self.last_bot_activity = {}
        self.last_user_engagement = {}
        self.proactive_dry_run = os.getenv('PROACTIVE_DRY_RUN', 'false').lower() == 'true'

        self.activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="the server | !help"
        )

    # ---------------- lifecycle ----------------

    async def setup_hook(self):
        logger.info("🚀 Initializing DaddyClintBot engine...")
        try:
            self.engine = DaddyClintBot()
            logger.info("✅ Engine loaded!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize engine: {e}")
            raise

        self.status_update.start()
        self.refresh_server_intel.start()
        self.daily_prune.start()
        self.ollama_watchdog.start()

        # discord.py 2.7 does NOT auto-register class-body @commands.command
        # methods on Bot subclasses — bind and register them explicitly.
        for name in ('status', 'health', 'news', 'vibe', 'stats',
                     'channels', 'persona', 'forgetme',
                     'reloadknowledge', 'proactive_status', 'snapshot'):
            cmd = getattr(type(self), name, None)
            if cmd is None:
                logger.warning(f"⚠️ Command method not found: {name}")
                continue
            if self.get_command(cmd.name) is not None:
                continue  # already registered
            cmd._callback = cmd.callback.__get__(self)
            self.add_command(cmd)
        logger.info(f"✅ Registered commands: {sorted(self.all_commands)}")

    async def on_ready(self):
        logger.info(f'✅ {self.user} connected — {len(self.guilds)} guild(s)')
        await self.change_presence(activity=self.activity)
        await self._refresh_server_intel()

    async def on_error(self, event_method, *args, **kwargs):
        # One broken event handler must never take the bot down
        logger.error(f"❌ Unhandled error in {event_method}", exc_info=True)

    async def on_member_join(self, member: discord.Member):
        """Track new faces so vibe reports can mention growth."""
        try:
            if self.engine:
                self.engine.db.log_channel_activity(
                    str(member.guild.id), 'server', member.display_name, '[JOIN]'
                )
        except Exception as e:
            logger.debug(f"Join log skipped: {e}")

    # ---------------- message handling ----------------

    async def on_message(self, message):
        # Ignore ALL bots (prevents bot-on-bot loops)
        if message.author.bot:
            return

        # Commands run first and exclusively
        prefix = os.getenv('BOT_PREFIX', '!')
        if message.content.startswith(prefix):
            await self.process_commands(message)
            return

        # Passive awareness: record guild chatter for !news
        if isinstance(message.channel, discord.TextChannel):
            self._log_activity(message)

        if isinstance(message.channel, discord.DMChannel):
            await self._handle_dm(message)
        elif self.user in message.mentions:
            await self._handle_mention(message)
        elif isinstance(message.channel, discord.TextChannel) and self.proactive_enabled:
            await self._evaluate_proactive_engagement(message)

    def _log_activity(self, message: discord.Message):
        """Record messages so the bot can report what's been happening."""
        try:
            content = message.content.strip()
            if not content or content.startswith(('http', '!', '?')):
                return
            if message.channel.name in self.blocked_channels:
                return
            if len(content) < 3:
                return
            self.engine.db.log_channel_activity(
                str(message.channel.id), message.channel.name,
                message.author.display_name, content
            )
        except Exception as e:
            logger.debug(f"Activity log skipped: {e}")

    def _is_owner(self, author) -> bool:
        return self.owner_id and str(author.id) == self.owner_id

    def _parse_thinkhard(self, author, content: str) -> tuple[str, int | None]:
        """Strip opt-in #thinkhard directive from content.

        Returns (cleaned_content, num_predict_override).
        - Owner: parses '#thinkhard', '#thinkhard 1500', '#thinkhard off'
                 and stores the override in self.owner_token_overrides
                 for this user, scoped to the bot's lifetime.
        - Non-owner: returns content unchanged, override=None (silently
          ignored — they get the normal short-reply budget).

        Examples (owner only):
          '#thinkhard what features do you have?'
              -> ('what features do you have?', 1000)
          '#thinkhard 1500 tell me about server config'
              -> ('tell me about server config', 1500)
          '#thinkhard off cool story'
              -> ('cool story', None)   # falls back to owner default
        """
        match = re.match(r'^\s*#thinkhard(?:\s+(\d+|off))?\b\s*(.*)$',
                         content, re.IGNORECASE | re.DOTALL)
        if not match:
            return content, None
        if not self._is_owner(author):
            # Non-owner tried the magic word — strip it silently so it
            # doesn't leak into the prompt, but don't enable big budget.
            return match.group(2).strip(), None

        spec = (match.group(1) or '').lower()
        rest = match.group(2).strip()
        user_id = str(author.id)

        if spec == 'off':
            self.owner_token_overrides.pop(user_id, None)
            logger.info(f"🧠 #thinkhard OFF for {author}")
            return rest, None  # owner default still applies

        if spec.isdigit():
            budget = max(100, min(int(spec), 4000))
        else:
            budget = self.DEFAULT_THINKHARD_BUDGET

        self.owner_token_overrides[user_id] = budget
        logger.info(f"🧠 #thinkhard ON ({budget} tokens) for {author}")
        return rest, budget

    async def _handle_dm(self, message: discord.Message):
        user_id = str(message.author.id)
        content = message.content.strip()

        if not content or content.startswith('http'):
            return

        # Opt-in #thinkhard (owner-only budget boost)
        content, np_override = self._parse_thinkhard(message.author, content)
        if np_override:
            await message.reply(f"🧠 #thinkhard ON — using {np_override} tokens. ask away.")
            if not content:
                return
        elif not content:
            # Non-owner typed '#thinkhard' alone — strip silently, no-op
            return

        logger.info(f"📨 DM from {message.author}: {content[:50]}...")
        typing_task = asyncio.create_task(self._keep_typing(message.channel))

        try:
            response, debug = await self.engine.process_message(
                user_id, message.author.display_name, content,
                is_owner=self._is_owner(message.author),
                num_predict_override=np_override,
            )
            logger.info(f"🤖 [{debug['intent']}] {response[:60]}...")
            await self._human_typing_delay(response)
            await message.reply(response)
        except Exception as e:
            logger.error(f"❌ DM error: {e}", exc_info=True)
            await message.reply("my brain glitched, try that again? 💀")
        finally:
            typing_task.cancel()

    async def _handle_mention(self, message: discord.Message):
        content = (message.content
                   .replace(f'<@{self.user.id}>', '')
                   .replace(f'<@!{self.user.id}>', '')
                   .strip())

        if not content:
            await message.reply("what's up? 👀")
            return
        if content.startswith('http'):
            return

        # Opt-in #thinkhard (owner-only budget boost)
        content, np_override = self._parse_thinkhard(message.author, content)
        if np_override:
            await message.reply(f"🧠 #thinkhard ON — using {np_override} tokens. ask away.")
            if not content:
                return
        elif not content:
            return  # non-owner used bare #thinkhard, strip silently

        user_id = str(message.author.id)
        logger.info(f"📣 Mention from {message.author}: {content[:50]}...")
        typing_task = asyncio.create_task(self._keep_typing(message.channel))

        try:
            response, debug = await self.engine.process_message(
                user_id, message.author.display_name, content,
                is_owner=self._is_owner(message.author),
                num_predict_override=np_override,
            )
            logger.info(f"🤖 [{debug['intent']}] {response[:60]}...")
            await self._human_typing_delay(response)
            await message.reply(response)
        except Exception as e:
            logger.error(f"❌ Mention error: {e}", exc_info=True)
            await message.reply("got distracted. what were we talking about? 🤔")
        finally:
            typing_task.cancel()

    async def _keep_typing(self, channel):
        while True:
            try:
                await channel.trigger_typing()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

    @staticmethod
    async def _human_typing_delay(response: str):
        """Simulate human typing pace so replies don't look instant/pasted.

        Floor of 0.6s (so short replies don't feel robotic) and cap of 2.5s
        (so long replies don't feel laggy). Mid-range scales at ~35ms/char,
        roughly matching a casual texter on mobile.
        """
        delay = max(0.6, min(2.5, len(response) * 0.035))
        await asyncio.sleep(delay)

    # ---------------- server awareness ----------------

    def _build_channel_directory(self):
        """Compact channel map injected into help-mode prompts."""
        lines = []
        for guild in self.guilds:
            lines.append(f"Server: {guild.name}")
            channels = [c for c in guild.text_channels
                        if c.name not in self.blocked_channels]
            for channel in channels[:40]:
                topic = (channel.topic or '').strip()
                topic = f" — {topic[:60]}" if topic else ""
                category = f"[{channel.category.name}] " if channel.category else ""
                lines.append(f"  {category}#{channel.name}{topic}")
            if len(channels) > 40:
                lines.append(f"  ...and {len(channels) - 40} more channels")
        directory = "\n".join(lines)
        if self.engine:
            self.engine.channel_directory = directory
        logger.info(f"🗂️  Channel directory built ({len(channels)} channels)")
        return directory

    # Channels whose pins/history likely hold rules and key info
    PRIORITY_CHANNEL_RE = re.compile(
        r'rules?|guidelines?|info|faq|welcome|announce|start|about|read.?first|how.?to',
        re.IGNORECASE
    )

    async def _gather_server_intel(self):
        """Auto-harvest the server's governing info from Discord itself:
        server description, roles, the rules channel, and pinned messages in
        rules/info/FAQ-style channels. Small-model budget: ~2500 chars."""
        sections = []
        for guild in self.guilds:
            if guild.description:
                sections.append(f"Server description: {guild.description}")

            role_names = [r.name for r in guild.roles
                          if not r.is_default() and not r.managed][:25]
            if role_names:
                sections.append("Roles: " + ", ".join(role_names))

            # Designated rules channel (Community servers)
            rules_ch = guild.rules_channel
            if rules_ch:
                try:
                    lines = []
                    async for m in rules_ch.history(limit=8):
                        if m.content.strip():
                            lines.append(m.content.strip()[:300])
                    if lines:
                        sections.append("From the rules channel:\n" + "\n".join(reversed(lines)))
                except discord.Forbidden:
                    logger.warning(f"⚠️ No access to rules channel in {guild.name} — "
                                   "grant Read Message History to use it")

            # Pinned messages in priority channels (rules/info/faq/etc.)
            for ch in guild.text_channels:
                if ch.name in self.blocked_channels:
                    continue
                if rules_ch and ch.id == rules_ch.id:
                    continue  # already covered above
                if not self.PRIORITY_CHANNEL_RE.search(ch.name):
                    continue
                try:
                    pins = await ch.pins()
                except discord.Forbidden:
                    continue
                except Exception as e:
                    logger.debug(f"Pin fetch failed for #{ch.name}: {e}")
                    continue
                lines = [p.content.strip()[:250] for p in pins[:5] if p.content.strip()]
                if lines:
                    sections.append(f"Pinned in #{ch.name}:\n" + "\n".join(lines))

        intel = "\n\n".join(sections)
        if len(intel) > 2500:
            intel = intel[:2500] + "\n...(truncated)"
        if self.engine:
            self.engine.auto_knowledge = intel
        logger.info(f"🧠 Server intel gathered ({len(intel)} chars)")
        return intel

    async def _refresh_server_intel(self):
        """Rebuild both the channel directory and the gathered server knowledge."""
        self._build_channel_directory()
        await self._gather_server_intel()

    # ---------------- proactive engagement ----------------

    async def _evaluate_proactive_engagement(self, message: discord.Message):
        try:
            if len(message.content) < self.min_message_length:
                return
            content_lower = message.content.lower().strip()
            if content_lower.startswith(('http', '!', '?')):
                return
            if len(message.content) < 20 and not any(c.isalpha() for c in message.content):
                return

            channel_id = str(message.channel.id)
            user_id = str(message.author.id)
            current_time = time.time()

            if message.channel.name in self.blocked_channels:
                return
            if self.allowed_channels and message.channel.name not in self.allowed_channels:
                return
            if current_time - self.last_bot_activity.get(channel_id, 0) < self.channel_cooldown_seconds:
                return
            if current_time - self.last_user_engagement.get(user_id, 0) < self.user_cooldown_seconds:
                return
            if self.owner_id and user_id == self.owner_id:
                return

            analysis = self.engine.analyzer.analyze(message.content)
            engagement_score = self._calculate_engagement_score(message, analysis)

            logger.info(f"📊 Proactive score: {engagement_score:.2f} "
                        f"(threshold: {self.proactive_threshold}) | {message.content[:40]}...")

            if engagement_score < self.proactive_threshold:
                return
            if self.proactive_dry_run:
                logger.info(f"🧪 [DRY RUN] Would respond to: {message.content[:50]}...")
                return

            logger.info(f"🎯 Proactive engagement (score: {engagement_score:.2f}) for {message.author}")
            self.last_bot_activity[channel_id] = current_time
            self.last_user_engagement[user_id] = current_time

            typing_task = asyncio.create_task(self._keep_typing(message.channel))
            try:
                directive = (
                    "MODE: PROACTIVE. You saw this message in the channel and are jumping "
                    "in uninvited (but naturally). Reference what they said specifically. "
                    "If they're frustrated, roast gently but validate. If celebrating, amplify. "
                    "If vulnerable, be supportive without being cringe."
                )
                response, _ = await self.engine.process_message(
                    user_id, message.author.display_name, message.content,
                    is_owner=False, force_intent='chat', extra_directive=directive
                )
                await self._human_typing_delay(response)
                await message.reply(response)
                logger.info(f"✅ Proactive response sent to {message.author}")
            except Exception as e:
                logger.error(f"❌ Proactive response error: {e}", exc_info=True)
            finally:
                typing_task.cancel()

        except Exception as e:
            logger.error(f"❌ Proactive evaluation error: {e}", exc_info=True)

    def _calculate_engagement_score(self, message: discord.Message, analysis: dict) -> float:
        score = 0.0
        content_lower = message.content.lower()
        sentiment = analysis.get('compound_score', 0)

        if sentiment < -0.5:
            score += 0.3
        elif sentiment > 0.5:
            score += 0.25

        if analysis.get('vulnerability_score', 0) > 0.6:
            score += 0.25
        if analysis.get('acr_trigger', False):
            score += 0.2

        topic_matches = sum(
            1 for keywords in self.topic_keywords.values()
            for keyword in keywords if keyword.lower() in content_lower
        )
        score += min(topic_matches * 0.08, 0.25)

        hour = message.created_at.hour if message.created_at else 12
        if 0 <= hour <= 3:
            score += 0.05
        if '?' in message.content:
            score += 0.1

        return min(score, 1.0)

    # ---------------- commands ----------------

    @commands.command(name='status')
    async def status(self, ctx):
        """Show bot status"""
        embed = discord.Embed(title="😏 DaddyClintBot Status", color=discord.Color.blue())
        embed.add_field(name="Engine", value="✅ Online" if self.engine else "❌ Offline", inline=True)
        embed.add_field(name="Latency", value=f"{round(self.latency * 1000)}ms", inline=True)
        embed.add_field(name="Servers", value=len(self.guilds), inline=True)
        await ctx.send(embed=embed)

    @commands.command(name='health')
    async def health(self, ctx):
        """Deep health check: Ollama, DB, uptime"""
        uptime = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, _ = divmod(remainder, 60)

        ollama_ok = self.engine.llm.check_connection() if self.engine else False
        embed = discord.Embed(
            title="🩺 Health Check",
            color=discord.Color.green() if ollama_ok else discord.Color.orange()
        )
        embed.add_field(name="Discord", value="✅ Connected", inline=True)
        embed.add_field(
            name="Ollama",
            value=f"✅ {self.engine.llm.model}" if ollama_ok else "❌ Unreachable",
            inline=True
        )
        embed.add_field(name="Uptime", value=f"{hours}h {minutes}m", inline=True)
        embed.add_field(name="Generations", value=self.engine.llm.total_generations, inline=True)
        embed.add_field(name="LLM Failures", value=self.engine.llm.total_failures, inline=True)
        if self.engine.llm.last_latency:
            embed.add_field(
                name="Last Gen",
                value=f"{self.engine.llm.last_latency:.1f}s",
                inline=True
            )
        embed.add_field(
            name="Activity (24h)",
            value=f"{self.engine.db.count_activity(24)} messages seen",
            inline=True
        )
        await ctx.send(embed=embed)

    @commands.command(name='news', aliases=['catchup', 'recap'])
    async def news(self, ctx):
        """Catch up on what's been happening across the server"""
        async with ctx.typing():
            response, _ = await self.engine.process_message(
                str(ctx.author.id), ctx.author.display_name,
                "Give me the news — what's been happening in the server?",
                is_owner=self._is_owner(ctx.author), force_intent='news'
            )
        # Digests can be long — split across the 2000-char limit
        for chunk in self._chunk(response):
            await ctx.send(chunk)

    @commands.command(name='vibe', aliases=['vibecheck', 'vibereport', 'pulse'])
    async def vibe(self, ctx):
        """Eyes-in-the-sky vibe report. Owner gets the full Machine-style
        intel (names, quotes, watch-outs); everyone else gets a light,
        aggregate-only vibe check."""
        is_owner = self._is_owner(ctx.author)
        async with ctx.typing():
            response, _ = await self.engine.process_message(
                str(ctx.author.id), ctx.author.display_name,
                "Give me the vibe report — how is everyone treating the server?",
                is_owner=is_owner, force_intent='vibe'
            )
        for chunk in self._chunk(response):
            await ctx.send(chunk)

    @commands.command(name='stats')
    async def stats(self, ctx):
        """Hard numbers, no AI: server-wide + current channel activity."""
        hours = self.engine.vibe_lookback_hours
        stats = self.engine.db.get_channel_stats(
            hours=hours, exclude_channels=self.blocked_channels)
        total = sum(n for _, n, _ in stats)
        joins = self.engine.db.get_join_count(hours)

        embed = discord.Embed(
            title=f"📈 Server Stats — last {hours}h",
            color=discord.Color.teal()
        )
        embed.add_field(name="Messages", value=total, inline=True)
        embed.add_field(name="Active Channels", value=len(stats), inline=True)
        embed.add_field(name="New Members", value=joins, inline=True)

        if stats:
            top = "\n".join(f"**#{name}** — {n} msgs / {a} people"
                            for name, n, a in stats[:5])
            embed.add_field(name="Busiest Channels", value=top, inline=False)

        # Current channel context
        if isinstance(ctx.channel, discord.TextChannel):
            here = next(((n, a) for name, n, a in stats
                         if name == ctx.channel.name), None)
            if here:
                embed.add_field(
                    name=f"This channel (#{ctx.channel.name})",
                    value=f"{here[0]} msgs from {here[1]} people in the window",
                    inline=False
                )

        # Per-user detail is owner-only (responsibility boundary)
        if self._is_owner(ctx.author):
            contributors = self.engine.db.get_top_contributors(
                hours=hours, exclude_channels=self.blocked_channels)
            if contributors:
                embed.add_field(
                    name="Top Contributors (owner only)",
                    value="\n".join(f"{a} — {n} msgs" for a, n in contributors),
                    inline=False
                )
        await ctx.send(embed=embed)

    @commands.command(name='snapshot', aliases=['snap', 'whatsgoingon'])
    async def snapshot(self, ctx, *, args: str = ''):
        """📸 On-demand server snapshot (pull-only, aggregates only)."""
        if ctx.guild is None:
            await ctx.reply("run that in the server 🙂")
            return
        typing_task = asyncio.create_task(self._keep_typing(ctx.channel))
        try:
            outcome = await run_snapshot(
                guild=ctx.guild, author=ctx.author,
                is_owner=self._is_owner(ctx.author), raw_args=args,
                engine=self.engine,
            )
            if outcome.embed is not None:
                try:
                    await ctx.send(embed=outcome.embed)
                    return
                except discord.HTTPException:
                    logger.warning("⚠️ snapshot embed rejected; sending plain text")
            for chunk in self._chunk(outcome.plain_text):
                await ctx.send(chunk)
        except Exception as e:
            logger.error(f"❌ snapshot command error: {e}", exc_info=True)
            await ctx.send("snapshot glitched on me — try again in a sec 💀")
        finally:
            typing_task.cancel()

    @commands.command(name='channels')
    async def channels(self, ctx):
        """List the server's channels and what they're for"""
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send("Ask me in the server 🙂")
            return

        embed = discord.Embed(
            title=f"🗺️ {ctx.guild.name} — Channel Map",
            color=discord.Color.blurple()
        )
        by_category = {}
        for channel in ctx.guild.text_channels:
            if channel.name in self.blocked_channels:
                continue
            category = channel.category.name if channel.category else "Other"
            topic = (channel.topic or "").strip()
            by_category.setdefault(category, []).append(
                f"**#{channel.name}**{f' — {topic[:70]}' if topic else ''}"
            )
        for category, lines in list(by_category.items())[:10]:
            embed.add_field(name=category, value="\n".join(lines[:12])[:1024], inline=False)
        await ctx.send(embed=embed)

    @commands.command(name='persona')
    async def persona(self, ctx):
        """Show bot persona info"""
        embed = discord.Embed(
            title="😏 Who is DaddyClintBot?",
            description="The server's resident funny guy who actually knows where everything is",
            color=discord.Color.purple()
        )
        embed.add_field(
            name="Vibe",
            value="Playful, sarcastic, genuinely helpful when it counts",
            inline=False
        )
        embed.add_field(
            name="What I do",
            value=("• Chat like a real person (DM or @mention me)\n"
                   "• Answer questions about the server (`!channels`, or just ask)\n"
                   "• Catch you up on what you missed (`!news`)\n"
                   "• Read the room (`!vibe`) and drop the numbers (`!stats`)\n"
                   "• Remember stuff about you (say `!forgetme` to wipe it)"),
            inline=False
        )
        await ctx.send(embed=embed)

    @commands.command(name='forgetme')
    async def forgetme(self, ctx):
        """Delete everything the bot remembers about you"""
        self.engine.db.clear_user_data(str(ctx.author.id))
        await ctx.reply("Done — wiped. You're a stranger to me now 😢")

    @commands.command(name='reloadknowledge')
    async def reloadknowledge(self, ctx):
        """(Owner only) Reload server_knowledge.md + re-gather rules/pins from Discord"""
        if not self._is_owner(ctx.author):
            await ctx.reply("Owner only, sorry 🤫")
            return
        self.engine.knowledge.reload()
        await self._refresh_server_intel()
        await ctx.reply("📖 Knowledge reloaded + re-gathered rules/pins from the server.")

    @commands.command(name='proactive')
    async def proactive_status(self, ctx):
        """Show proactive engagement status"""
        embed = discord.Embed(
            title="🎯 Proactive Engagement",
            color=discord.Color.green() if self.proactive_enabled else discord.Color.red()
        )
        embed.add_field(name="Status", value="✅ Enabled" if self.proactive_enabled else "❌ Disabled", inline=True)
        embed.add_field(name="Threshold", value=f"{self.proactive_threshold:.2f}", inline=True)
        embed.add_field(name="Dry Run", value="✅" if self.proactive_dry_run else "❌", inline=True)
        embed.add_field(name="Channel Cooldown", value=f"{self.channel_cooldown_seconds}s", inline=True)
        embed.add_field(name="User Cooldown", value=f"{self.user_cooldown_seconds}s", inline=True)
        if self.allowed_channels:
            embed.add_field(name="Allowed", value=", ".join(self.allowed_channels), inline=False)
        embed.add_field(name="Blocked", value=", ".join(self.blocked_channels), inline=False)
        await ctx.send(embed=embed)

    # ---------------- background loops ----------------

    @tasks.loop(minutes=5)
    async def status_update(self):
        if self.engine:
            await self.change_presence(activity=self.activity)

    @tasks.loop(minutes=30)
    async def refresh_server_intel(self):
        """Channels, rules and pins change; rebuild the knowledge periodically."""
        try:
            await self._refresh_server_intel()
        except Exception as e:
            logger.error(f"❌ Intel refresh failed: {e}")

    @tasks.loop(hours=24)
    async def daily_prune(self):
        try:
            self.engine.db.prune_old_data()
        except Exception as e:
            logger.error(f"❌ Prune failed: {e}")

    @tasks.loop(minutes=5)
    async def ollama_watchdog(self):
        """Log Ollama up/down transitions so outages are visible."""
        if not self.engine:
            return
        ok = self.engine.llm.check_connection()
        was_ok = getattr(self, '_ollama_was_ok', True)
        if ok and not was_ok:
            logger.info("✅ Ollama is back online")
        elif not ok and was_ok:
            logger.error("🚨 Ollama went offline — bot will use fallback replies until it returns")
        self._ollama_was_ok = ok

    @status_update.before_loop
    @refresh_server_intel.before_loop
    @daily_prune.before_loop
    @ollama_watchdog.before_loop
    async def before_loops(self):
        await self.wait_until_ready()

    @staticmethod
    def _chunk(text: str, size: int = 1990):
        for i in range(0, len(text), size):
            yield text[i:i + size]


async def main():
    """Resilient entry point: rebuild the bot and reconnect on fatal errors."""
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("❌ DISCORD_TOKEN not found! Set it in your .env file")
        sys.exit(1)

    attempt = 0
    while True:
        bot = DaddyClintDiscordBot()

        # Graceful shutdown on SIGTERM (systemd stop)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, lambda: asyncio.create_task(bot.close())
                )
            except (NotImplementedError, RuntimeError):
                pass  # Windows / restricted loops

        try:
            await bot.start(token)
            logger.info("👋 Bot closed cleanly; not restarting.")
            return
        except discord.LoginFailure:
            logger.error("❌ Invalid DISCORD_TOKEN — fix your .env and restart.")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down gracefully...")
            await bot.close()
            return
        except Exception as e:
            attempt += 1
            delay = min(10 * attempt, 300)
            logger.error(
                f"❌ Fatal error ({e}); restarting fresh in {delay}s "
                f"(attempt {attempt})", exc_info=True
            )
            try:
                await bot.close()
            except Exception:
                pass
            await asyncio.sleep(delay)


if __name__ == '__main__':
    asyncio.run(main())
