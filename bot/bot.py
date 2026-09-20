import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import html
import json
import logging
import asyncio
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List

import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction, ChatType

import config
import database
import openai_utils
from memory import MemoryEngine
from multimodal import MultimodalInterpreter
from personality import PersonalityEngine
from group_engine import GroupEngine
from context_engine import ConversationContextEngine

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Core singletons
db = database.Database()
memory_engine = MemoryEngine(db)
multimodal = MultimodalInterpreter(db)
personality = PersonalityEngine(db)
group_engine = GroupEngine(db)
context_engine = ConversationContextEngine(db)

# Locks and active tasks
semaphores: Dict[int, asyncio.Semaphore] = {}
active_tasks: Dict[int, asyncio.Task] = {}
seen_updates = set()

HELP_MESSAGE = """🤖 <b>Personal AI Assistant</b>

You can talk to me completely naturally — no special commands required:
• <b>Ask anything:</b> Coding, brainstorming, analysis, writing, or discussion.
• <b>Remember facts:</b> Say <i>"Remember that my project is called Aurora"</i>.
• <b>Forget facts:</b> Say <i>"Forget my project name"</i>.
• <b>View memories:</b> Ask <i>"What do you remember about me?"</i>.
• <b>Off the record:</b> Say <i>"Don't remember this"</i> to keep a turn private.
• <b>Multimodal:</b> Send voice messages, photos, or documents (PDF, TXT, code).

👥 <b>In Group Chats:</b>
• Mention me (@{bot_username}) or reply to my messages to chat.
• Ask <i>"What did I miss?"</i> for a quick digest of recent discussions.
"""

OWNER_HELP = """👑 <b>Owner Controls:</b>
⚪ /panel – Open the Owner Control Panel
⚪ /sethome – Designate current group as Home Chat
⚪ /broadcast &lt;text&gt; – Announcement to all users & groups
"""


def get_semaphore(entity_id: int) -> asyncio.Semaphore:
    if entity_id not in semaphores:
        semaphores[entity_id] = asyncio.Semaphore(1)
    return semaphores[entity_id]


def get_entity_id(update: Update) -> int:
    if update.effective_chat and update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        return update.effective_chat.id
    return update.effective_user.id


def is_owner(user_id: int) -> bool:
    if config.owner_id is None:
        return False
    return int(user_id) == int(config.owner_id)


def split_text_into_chunks(text: str, chunk_size: int = 4000) -> List[str]:
    chunks = []
    current = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > chunk_size:
            chunks.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks or [text]


async def register_chat_and_user(update: Update, context: CallbackContext):
    user = update.effective_user
    chat = update.effective_chat

    if user:
        if not db.check_if_user_exists(user.id):
            db.add_new_user(
                user_id=user.id,
                chat_id=chat.id if chat else user.id,
                username=user.username or "",
                first_name=user.first_name or "",
                last_name=user.last_name or ""
            )
            db.start_new_dialog(user.id)
        else:
            db.set_user_attribute(user.id, "last_interaction", datetime.now())

    if chat and chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        db.add_or_update_chat(
            chat.id,
            title=chat.title or "Group",
            chat_type=chat.type,
            username=chat.username
        )
        if db.get_chat_attribute(chat.id, "current_dialog_id") is None:
            db.start_new_dialog(chat.id)


async def should_respond_in_group(update: Update, context: CallbackContext) -> bool:
    message = update.message
    if not message:
        return False

    chat = update.effective_chat
    if not chat or chat.type == ChatType.PRIVATE:
        return True

    # 1. Replied to bot
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == context.bot.id:
            return True

    # 2. Mentioned bot username
    bot_username = context.bot.username or ""
    text_content = (message.text or message.caption or "").lower()
    if bot_username and f"@{bot_username.lower()}" in text_content:
        return True

    # 3. Check Home Chat
    home_chat_id = db.get_global_setting("home_chat_id", None)
    if home_chat_id is not None and int(chat.id) == int(home_chat_id):
        return db.get_global_setting("home_auto_reply", True)

    # 4. Check Group Auto-Reply setting
    if db.get_chat_attribute(chat.id, "auto_reply", False):
        return True

    return False


# =====================================================================
# UNIFIED NATURAL CONVERSATION PIPELINE
# =====================================================================

async def process_user_turn(
    update: Update,
    context: CallbackContext,
    raw_text: str,
    image_data_url: Optional[str] = None,
    document_context: Optional[str] = None,
    initial_status: str = "Thinking…"
):
    user = update.effective_user
    chat = update.effective_chat
    entity_id = get_entity_id(update)
    is_group = chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]
    scope = f"group:{entity_id}" if is_group else f"user:{entity_id}"

    # Strip bot mention in groups
    clean_text = raw_text
    if is_group and context.bot.username:
        clean_text = clean_text.replace(f"@{context.bot.username}", "").strip()

    # 1. Natural Language Memory Intent Check
    intent, intent_arg = memory_engine.detect_memory_intent(clean_text)
    is_off_the_record = (intent == "off_the_record")

    if intent == "show":
        mems = db.get_active_memories(scope, entity_id)
        if not mems:
            await update.message.reply_text("I don't have any saved memories for you yet. Just tell me what you'd like me to remember!")
            return
        lines = ["🧠 <b>Here is what I remember:</b>\n"]
        for m in mems[:15]:
            lines.append(f"• {html.escape(m.get('content', ''))}")
        lines.append("\n<i>You can ask me to forget any of these anytime!</i>")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        return

    elif intent == "forget":
        target = intent_arg or clean_text
        count = db.delete_memories_by_keyword(scope, entity_id, target)
        if count > 0:
            await update.message.reply_text(f"✅ I've forgotten what I knew about <b>{html.escape(target)}</b>.", parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(f"I couldn't find any active memories matching <i>'{html.escape(target)}'</i>.", parse_mode=ParseMode.HTML)
        return

    elif intent == "explicit_remember":
        fact = intent_arg or clean_text
        db.add_memory(scope=scope, entity_id=entity_id, content=fact, category="personal", importance=0.8)
        await update.message.reply_text(f"Got it! I'll remember that: <i>{html.escape(fact)}</i>", parse_mode=ParseMode.HTML)
        return

    elif intent == "memory_toggle":
        if intent_arg == "off":
            db.set_user_attribute(user.id, "memory_active", False)
            await update.message.reply_text("Understood. I will no longer remember new information from our conversations.")
        else:
            db.set_user_attribute(user.id, "memory_active", True)
            await update.message.reply_text("Memory resumed! I will remember relevant context from now on.")
        return

    # 2. Natural Group Digest Check ("What did I miss?")
    if is_group and group_engine.is_digest_request(clean_text):
        await update.message.chat.send_action(action=ChatAction.TYPING)
        digest = await group_engine.generate_group_digest(chat.id, requester_name=user.first_name if user else "")
        await update.message.reply_text(digest, parse_mode=ParseMode.HTML)
        return

    # 3. Resolve Reply Context
    reply_context = None
    if update.message.reply_to_message:
        replied = update.message.reply_to_message
        reply_sender = replied.from_user.first_name if replied.from_user else "User"
        replied_text = replied.text or replied.caption or ""
        if replied_text:
            reply_context = f"{reply_sender}: {replied_text[:200]}"

    # 4. Check for Image Follow-up ("make it darker", "what is this picture")
    if not image_data_url:
        image_data_url = multimodal.resolve_conversational_image(entity_id, clean_text)

    # 5. Build LLM Messages via ConversationContextEngine
    user_display_name = user.first_name if user else "User"
    formatted_user_text = clean_text
    if is_group:
        formatted_user_text = group_engine.format_group_message_context(
            sender_name=user_display_name,
            text=clean_text,
            reply_to_sender=update.message.reply_to_message.from_user.first_name if update.message.reply_to_message and update.message.reply_to_message.from_user else None,
            reply_to_text=update.message.reply_to_message.text if update.message.reply_to_message else None
        )

    llm_messages = await context_engine.build_llm_messages(
        entity_id=entity_id,
        current_user_text=formatted_user_text,
        user_name=user_display_name,
        is_group=is_group,
        group_title=chat.title if is_group else "",
        image_data_url=image_data_url,
        document_context=document_context,
        reply_context=reply_context
    )

    # 6. Lock Semaphore for this Entity
    sem = get_semaphore(entity_id)
    if sem.locked():
        await update.message.reply_text("⏳ Working on your previous request, please wait a moment…", reply_to_message_id=update.message.id)
        return

    current_model = (
        db.get_user_attribute(entity_id, "current_model")
        if entity_id > 0
        else db.get_chat_attribute(entity_id, "current_model", config.default_model)
    ) or config.default_model

    chatgpt_client = openai_utils.ChatGPT(model=current_model)

    async def execute_turn():
        # Send initial status
        placeholder = await update.message.reply_text(initial_status, reply_to_message_id=update.message.id)
        await update.message.chat.send_action(action=ChatAction.TYPING)

        answer = ""
        n_input_tokens = 0
        n_output_tokens = 0

        try:
            if config.enable_message_streaming:
                gen = chatgpt_client.send_message_stream(llm_messages)
                prev_text = ""
                last_edit_time = asyncio.get_event_loop().time()

                async for status, text_chunk, (n_in, n_out) in gen:
                    answer = text_chunk
                    n_input_tokens, n_output_tokens = n_in, n_out
                    now = asyncio.get_event_loop().time()

                    # Throttle edits to respect Telegram limits
                    if (abs(len(answer) - len(prev_text)) >= 70 or status == "finished") and (now - last_edit_time >= 0.4 or status == "finished"):
                        display_text = answer[:4000]
                        try:
                            await context.bot.edit_message_text(
                                display_text,
                                chat_id=placeholder.chat_id,
                                message_id=placeholder.message_id,
                                parse_mode=ParseMode.HTML
                            )
                        except telegram.error.BadRequest as e:
                            if "Message is not modified" not in str(e):
                                # Fallback to plain text if HTML tags broken
                                try:
                                    await context.bot.edit_message_text(
                                        display_text,
                                        chat_id=placeholder.chat_id,
                                        message_id=placeholder.message_id
                                    )
                                except Exception:
                                    pass
                        prev_text = answer
                        last_edit_time = now
            else:
                answer, (n_input_tokens, n_output_tokens) = await chatgpt_client.send_message(llm_messages)
                try:
                    await context.bot.edit_message_text(
                        answer[:4000],
                        chat_id=placeholder.chat_id,
                        message_id=placeholder.message_id,
                        parse_mode=ParseMode.HTML
                    )
                except Exception:
                    await context.bot.edit_message_text(
                        answer[:4000],
                        chat_id=placeholder.chat_id,
                        message_id=placeholder.message_id
                    )

            # Send remaining chunks if response exceeds Telegram 4096 character limit
            if len(answer) > 4000:
                for chunk in split_text_into_chunks(answer[4000:]):
                    try:
                        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
                    except Exception:
                        await update.message.reply_text(chunk)

            # Persist dialog message
            turn_record = {
                "user": formatted_user_text,
                "bot": answer,
                "date": datetime.now()
            }
            db.set_dialog_messages(entity_id, db.get_dialog_messages(entity_id) + [turn_record])
            db.update_n_used_tokens(entity_id, current_model, n_input_tokens, n_output_tokens)

            # Update conversational state for natural follow-ups ("make it shorter", "another version")
            db.set_conversational_state(entity_id, {
                "last_output": answer,
                "last_user_query": clean_text
            })

            # Silently adapt user communication style
            if entity_id > 0:
                personality.adapt_user_style_from_message(entity_id, clean_text)

            # Background Tasks: Memory Extraction & Rolling Summarization (Non-blocking!)
            if not is_off_the_record:
                is_mem_active = (
                    db.get_user_attribute(entity_id, "memory_active")
                    if entity_id > 0 else True
                )
                if is_mem_active and config.memory_enabled:
                    asyncio.create_task(
                        memory_engine.extract_and_update_memory(
                            scope=scope,
                            entity_id=entity_id,
                            user_message=clean_text,
                            bot_response=answer
                        )
                    )

            # Background rolling summarization if dialog grows long
            asyncio.create_task(context_engine.maybe_summarize_in_background(entity_id))

        except asyncio.CancelledError:
            await update.message.reply_text("🛑 Canceled.")
            raise
        except Exception as e:
            logger.error(f"Error in turn execution: {traceback.format_exc()}")
            friendly_err = "I couldn't complete that response right now. Please try again in a moment."
            try:
                await context.bot.edit_message_text(
                    friendly_err,
                    chat_id=placeholder.chat_id,
                    message_id=placeholder.message_id
                )
            except Exception:
                await update.message.reply_text(friendly_err)

    async with sem:
        task = asyncio.create_task(execute_turn())
        active_tasks[entity_id] = task
        try:
            await task
        finally:
            if entity_id in active_tasks:
                del active_tasks[entity_id]


# =====================================================================
# TELEGRAM HANDLERS
# =====================================================================

async def text_message_handle(update: Update, context: CallbackContext):
    if update.edited_message or not update.message or not update.message.text:
        return

    # Check duplicate update
    if update.update_id in seen_updates:
        return
    seen_updates.add(update.update_id)
    if len(seen_updates) > 5000:
        seen_updates.clear()

    await register_chat_and_user(update, context)

    # In groups: store message in digest buffer for 'What did I miss?'
    chat = update.effective_chat
    user = update.effective_user
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        db.append_group_message_buffer(
            chat.id,
            sender_name=user.first_name if user else "Member",
            sender_id=user.id if user else 0,
            text=update.message.text
        )
        if not await should_respond_in_group(update, context):
            return

    # Cancel trigger in natural language
    if update.message.text.strip().lower() in ["cancel", "stop", "nevermind", "abort"]:
        entity_id = get_entity_id(update)
        if entity_id in active_tasks:
            active_tasks[entity_id].cancel()
            await update.message.reply_text("🛑 Canceled.")
            return

    await process_user_turn(
        update=update,
        context=context,
        raw_text=update.message.text,
        initial_status="Thinking…"
    )


async def photo_message_handle(update: Update, context: CallbackContext):
    if update.edited_message or not update.message:
        return

    await register_chat_and_user(update, context)
    chat = update.effective_chat
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        if not await should_respond_in_group(update, context):
            return

    entity_id = get_entity_id(update)
    caption, image_url = await multimodal.process_photo(update, context, entity_id)

    await process_user_turn(
        update=update,
        context=context,
        raw_text=caption or "What is in this image?",
        image_data_url=image_url,
        initial_status="Inspecting image…"
    )


async def voice_message_handle(update: Update, context: CallbackContext):
    if not update.message or not update.message.voice:
        return

    await register_chat_and_user(update, context)
    chat = update.effective_chat
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        if not await should_respond_in_group(update, context):
            return

    status_msg = await update.message.reply_text("🎤 Transcribing voice…", reply_to_message_id=update.message.id)
    try:
        transcript, duration = await multimodal.process_voice(update, context)
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await status_msg.edit_text("Could not transcribe voice audio.")
        return

    if not transcript:
        await status_msg.edit_text("Could not detect any speech in the audio.")
        return

    await status_msg.delete()
    await process_user_turn(
        update=update,
        context=context,
        raw_text=transcript,
        initial_status="Thinking…"
    )


async def document_message_handle(update: Update, context: CallbackContext):
    if not update.message or not update.message.document:
        return

    await register_chat_and_user(update, context)
    chat = update.effective_chat
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        if not await should_respond_in_group(update, context):
            return

    status_msg = await update.message.reply_text("📄 Reading document…", reply_to_message_id=update.message.id)
    caption, doc_text = await multimodal.process_document(update, context)
    await status_msg.delete()

    await process_user_turn(
        update=update,
        context=context,
        raw_text=caption or "Please analyze this document.",
        document_context=doc_text,
        initial_status="Analyzing document…"
    )


# =====================================================================
# COMMANDS & OWNER PANEL
# =====================================================================

async def start_handle(update: Update, context: CallbackContext):
    await register_chat_and_user(update, context)
    user = update.effective_user
    chat = update.effective_chat

    bot_user = context.bot.username or "Bot"
    text = (
        f"Hey <b>{html.escape(user.first_name)}</b> — I'm your personal AI assistant.\n\n"
        "You can talk to me normally. I remember context across our conversations, adapt to how you like to work, "
        "and can read images, documents, and voice messages.\n\n"
        "What are we working on today?"
    )

    keyboard = []
    if chat.type == ChatType.PRIVATE and is_owner(user.id):
        keyboard.append([InlineKeyboardButton("👑 Owner Control Panel", callback_data="owner_panel|main")])

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)


async def help_handle(update: Update, context: CallbackContext):
    await register_chat_and_user(update, context)
    bot_user = context.bot.username or "Bot"
    text = HELP_MESSAGE.format(bot_username=bot_user)
    if is_owner(update.effective_user.id):
        text += "\n" + OWNER_HELP
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def sethome_handle(update: Update, context: CallbackContext):
    await register_chat_and_user(update, context)
    user = update.effective_user
    chat = update.effective_chat

    if not is_owner(user.id):
        await update.message.reply_text("⛔ Only the bot owner can configure the Home Chat.")
        return

    if chat.type == ChatType.PRIVATE:
        home_id = db.get_global_setting("home_chat_id", None)
        home_title = db.get_global_setting("home_chat_title", "None")
        await update.message.reply_text(
            f"🏠 Current Home Chat: <b>{html.escape(str(home_title))}</b> (ID: <code>{home_id}</code>)\n\n"
            "To set a new home chat, run <code>/sethome</code> directly inside the target group!",
            parse_mode=ParseMode.HTML
        )
        return

    db.set_global_setting("home_chat_id", chat.id)
    db.set_global_setting("home_chat_title", chat.title or "Home Group")
    db.set_global_setting("home_auto_reply", True)
    db.set_chat_attribute(chat.id, "auto_reply", True)

    text = (
        f"🏠 <b>Home Chat Activated!</b>\n\n"
        f"This group (<b>{html.escape(chat.title or 'Group')}</b>) is now configured as the Home Chat.\n"
        "✨ I will now actively chat and respond to all members here!"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_chat_and_user(update, context)
    entity_id = get_entity_id(update)
    db.start_new_dialog(entity_id)
    await update.message.reply_text("🔄 Started a fresh dialog. What's on your mind?")


async def cancel_handle(update: Update, context: CallbackContext):
    entity_id = get_entity_id(update)
    if entity_id in active_tasks:
        active_tasks[entity_id].cancel()
        await update.message.reply_text("🛑 Canceled.")
    else:
        await update.message.reply_text("Nothing active to cancel.")


# --- Owner Panel ---

def get_owner_panel_main():
    total_users = db.get_total_users_count()
    total_groups = db.get_total_groups_count()
    total_mems = db.get_total_memories_count()
    home_id = db.get_global_setting("home_chat_id", "Not set")
    home_title = db.get_global_setting("home_chat_title", "None")
    home_auto = db.get_global_setting("home_auto_reply", True)

    text = (
        "👑 <b>Owner Control Panel</b>\n\n"
        f"👥 <b>Users:</b> {total_users}\n"
        f"👥 <b>Groups:</b> {total_groups}\n"
        f"🧠 <b>Durable Memories:</b> {total_mems}\n"
        f"🏠 <b>Home Chat:</b> {html.escape(str(home_title))} (<code>{home_id}</code>)\n"
        f"⚡ <b>Home Auto-Reply:</b> {'✅ Active' if home_auto else '❌ Mentions Only'}\n"
        f"🤖 <b>Default Model:</b> <code>{config.default_model}</code>\n"
    )

    keyboard = [
        [
            InlineKeyboardButton("📊 Detailed Stats", callback_data="owner_panel|stats"),
            InlineKeyboardButton("🏠 Home Chat", callback_data="owner_panel|home")
        ],
        [
            InlineKeyboardButton("🤖 Switch Model", callback_data="owner_panel|models"),
            InlineKeyboardButton("📢 Broadcast", callback_data="owner_panel|broadcast_info")
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="owner_panel|main")
        ]
    ]
    return text, InlineKeyboardMarkup(keyboard)


async def panel_command_handle(update: Update, context: CallbackContext):
    user = update.effective_user
    if not is_owner(user.id):
        await update.message.reply_text("⛔ Access denied. Only the owner can access the control panel.")
        return

    if update.effective_chat.type != ChatType.PRIVATE:
        await update.message.reply_text("🔒 The control panel can only be accessed in private DMs with the bot.")
        return

    text, reply_markup = get_owner_panel_main()
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def owner_panel_callback_handle(update: Update, context: CallbackContext):
    query = update.callback_query
    user = query.from_user
    if not is_owner(user.id):
        await query.answer("⛔ Access denied.", show_alert=True)
        return

    await query.answer()
    action = query.data.split("|")[1]

    if action == "main":
        text, reply_markup = get_owner_panel_main()
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        except Exception:
            pass

    elif action == "stats":
        total_users = db.get_total_users_count()
        total_groups = db.get_total_groups_count()
        total_mems = db.get_total_memories_count()

        text = (
            "📊 <b>Detailed Bot Statistics</b>\n\n"
            f"• Registered Users: <b>{total_users}</b>\n"
            f"• Groups Connected: <b>{total_groups}</b>\n"
            f"• Stored Long-Term Memories: <b>{total_mems}</b>\n"
            f"• Active Model: <code>{config.default_model}</code>\n"
            f"• Memory Weight Semantic: <code>{config.memory_weight_semantic}</code>\n"
            f"• Memory Weight Importance: <code>{config.memory_weight_importance}</code>\n"
        )
        keyboard = [[InlineKeyboardButton("« Back", callback_data="owner_panel|main")]]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif action == "home":
        home_id = db.get_global_setting("home_chat_id", None)
        home_title = db.get_global_setting("home_chat_title", "None")
        home_auto = db.get_global_setting("home_auto_reply", True)

        text = (
            "🏠 <b>Home Chat Settings</b>\n\n"
            f"Current Home Chat: <b>{html.escape(str(home_title))}</b>\n"
            f"Chat ID: <code>{home_id or 'Not set'}</code>\n"
            f"Mode: <b>{'All Messages' if home_auto else 'Mentions & Replies Only'}</b>\n\n"
            "<i>To set a new home group, open that group and send /sethome.</i>"
        )
        toggle_label = "Switch to Mentions Only" if home_auto else "Switch to All Messages"
        keyboard = [
            [InlineKeyboardButton(toggle_label, callback_data="owner_panel|toggle_home_auto")],
            [InlineKeyboardButton("❌ Clear Home Chat", callback_data="owner_panel|clear_home")],
            [InlineKeyboardButton("« Back", callback_data="owner_panel|main")]
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    elif action == "toggle_home_auto":
        cur = db.get_global_setting("home_auto_reply", True)
        db.set_global_setting("home_auto_reply", not cur)
        query.data = "owner_panel|home"
        await owner_panel_callback_handle(update, context)

    elif action == "clear_home":
        db.set_global_setting("home_chat_id", None)
        db.set_global_setting("home_chat_title", "None")
        query.data = "owner_panel|home"
        await owner_panel_callback_handle(update, context)

    elif action == "models":
        text = "🤖 <b>Select Global Default Model</b>:"
        buttons = []
        for m in config.models.get("available_text_models", []):
            m_name = config.models["info"].get(m, {}).get("name", m)
            buttons.append([InlineKeyboardButton(m_name, callback_data=f"owner_panel|set_model|{m}")])
        buttons.append([InlineKeyboardButton("« Back", callback_data="owner_panel|main")])
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(buttons), parse_mode=ParseMode.HTML)

    elif action.startswith("set_model"):
        selected = query.data.split("|")[2]
        config.default_model = selected
        text, reply_markup = get_owner_panel_main()
        await query.edit_message_text(
            f"✅ <b>Default model switched to:</b> <code>{selected}</code>\n\n" + text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

    elif action == "broadcast_info":
        text = "📢 <b>Broadcast Announcement</b>\n\nSend to all users and groups via:\n<code>/broadcast Your message text here</code>"
        keyboard = [[InlineKeyboardButton("« Back", callback_data="owner_panel|main")]]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


async def broadcast_handle(update: Update, context: CallbackContext):
    user = update.effective_user
    if not is_owner(user.id):
        await update.message.reply_text("⛔ Only the owner can send broadcasts.")
        return

    broadcast_text = " ".join(context.args) if context.args else ""
    if not broadcast_text:
        await update.message.reply_text("Usage: <code>/broadcast Your message here</code>", parse_mode=ParseMode.HTML)
        return

    all_user_ids = db.get_all_user_ids()
    all_chats = [c["_id"] for c in db.get_all_chats() if c["_id"] < 0]
    targets = list(set(all_user_ids + all_chats))

    status_msg = await update.message.reply_text(f"🚀 Broadcasting to {len(targets)} recipients…")
    success = 0
    failed = 0

    for target_id in targets:
        try:
            await context.bot.send_message(
                chat_id=target_id,
                text=f"📢 <b>Announcement:</b>\n\n{broadcast_text}",
                parse_mode=ParseMode.HTML
            )
            success += 1
            await asyncio.sleep(0.04)
        except Exception:
            failed += 1

    await status_msg.edit_text(
        f"✅ <b>Broadcast Complete!</b>\n\n"
        f"• Delivered: <b>{success}</b>\n"
        f"• Failed/Blocked: <b>{failed}</b>",
        parse_mode=ParseMode.HTML
    )


async def error_handle(update: Update, context: CallbackContext):
    logger.error("Exception handling update:", exc_info=context.error)


async def post_init(application: Application):
    commands = [
        BotCommand("/new", "Start fresh dialog"),
        BotCommand("/help", "Show help message"),
    ]
    if config.owner_id:
        commands.append(BotCommand("/panel", "👑 Owner Control Panel"))
        commands.append(BotCommand("/sethome", "🏠 Set Home Chat"))

    await application.bot.set_my_commands(commands)


def run_bot() -> None:
    if not config.telegram_token:
        raise ValueError("TELEGRAM_TOKEN is required. Set it in config/config.yml or via environment variable.")

    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # Core commands
    application.add_handler(CommandHandler("start", start_handle))
    application.add_handler(CommandHandler("help", help_handle))
    application.add_handler(CommandHandler("new", new_dialog_handle))
    application.add_handler(CommandHandler("cancel", cancel_handle))

    # Owner commands
    application.add_handler(CommandHandler("panel", panel_command_handle))
    application.add_handler(CommandHandler("admin", panel_command_handle))
    application.add_handler(CommandHandler("sethome", sethome_handle))
    application.add_handler(CommandHandler("broadcast", broadcast_handle))
    application.add_handler(CallbackQueryHandler(owner_panel_callback_handle, pattern="^owner_panel"))

    # Multimodal Message Handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handle))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_message_handle))
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handle))
    application.add_handler(MessageHandler(filters.Document.ALL & ~filters.COMMAND, document_message_handle))

    application.add_error_handler(error_handle)

    logger.info("Mira-style Personal AI Bot starting up...")
    application.run_polling()


if __name__ == "__main__":
    run_bot()
