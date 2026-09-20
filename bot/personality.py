import json
from datetime import datetime
from typing import Dict, List, Optional, Any

BASE_ASSISTANT_IDENTITY = """You are a capable, intelligent personal AI assistant in Telegram.
Your communication style is direct, natural, and human-like.
CRITICAL BEHAVIORAL GUIDELINES:
1. NEVER introduce yourself with 'As an AI language model' or 'As an AI assistant'.
2. NEVER use generic corporate boilerplate disclaimers.
3. Keep simple answers crisp and direct (e.g. 'what is 2+2' -> '4.').
4. For complex requests, give thoughtful, well-structured, actionable answers without unnecessary fluff.
5. You have persistent memory across conversations. Refer to past context naturally without announcing 'According to my database'.
6. If the user refers to 'it', 'that', 'make it shorter', or 'give me another version', understand that they are referring to the active object or previous message.
7. Use Telegram-safe formatting (bold, italic, code blocks) cleanly and intentionally.
"""


class PersonalityEngine:
    def __init__(self, db):
        self.db = db

    def get_system_prompt(
        self,
        entity_id: int,
        chat_mode: str = "assistant",
        user_name: str = "",
        is_group: bool = False,
        group_title: str = "",
        user_timezone: Optional[str] = None
    ) -> str:
        parts = [BASE_ASSISTANT_IDENTITY]

        # Group vs Private identity rules
        if is_group:
            parts.append(
                f"You are speaking in a Telegram group chat named '{group_title or 'Group'}'.\n"
                "In group chats, multiple people participate. Messages from members are prefixed with '[Sender Name]: message'.\n"
                "Address members naturally. Do not lecture the group. Be a helpful, friendly participant."
            )
        else:
            if user_name:
                parts.append(f"You are speaking one-on-one with {user_name}.")

        # Time context
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        time_info = f"Current server time: {now_str}."
        if user_timezone:
            time_info += f" User timezone: {user_timezone}."
        parts.append(time_info)

        # Language preference
        if entity_id > 0:
            lang = self.db.get_user_attribute(entity_id, "language")
            if lang:
                lang_names = {
                    "en": "English", "ar": "Arabic", "es": "Spanish",
                    "fr": "French", "de": "German", "pt": "Portuguese",
                    "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
                    "ko": "Korean", "turkish": "Turkish", "fa": "Persian/Farsi",
                }
                lang_name = lang_names.get(lang, lang)
                parts.append(f"IMPORTANT: The user has set their preferred language to {lang_name}. Always respond in {lang_name}.")

        # Style preferences for user
        if entity_id > 0:
            prefs = self.db.get_user_attribute(entity_id, "style_preferences")
            if prefs and isinstance(prefs, dict):
                style_lines = ["User Communication Preferences:"]
                if prefs.get("verbosity") == "low":
                    style_lines.append("- Keep responses concise and direct.")
                elif prefs.get("verbosity") == "detailed":
                    style_lines.append("- Provide thorough, detailed explanations.")
                if prefs.get("formality") == "casual":
                    style_lines.append("- Use an easygoing, casual tone.")
                if prefs.get("technical_detail") == "high":
                    style_lines.append("- Provide rigorous technical depth.")
                if len(style_lines) > 1:
                    parts.append("\n".join(style_lines))

        return "\n\n".join(parts)

    def adapt_user_style_from_message(self, user_id: int, text: str):
        """Silently learns and adapts user communication style without verbose announcements."""
        if user_id <= 0:
            return

        lower = text.lower()
        current_prefs = self.db.get_user_attribute(user_id, "style_preferences") or {
            "verbosity": "balanced",
            "formality": "natural",
            "technical_detail": "adaptive"
        }

        updated = False
        if any(w in lower for w in ["be concise", "keep it short", "shorter", "too long", "too verbose", "brief answer", "bullet points only"]):
            current_prefs["verbosity"] = "low"
            updated = True
        elif any(w in lower for w in ["explain in detail", "elaborate", "give more detail", "in depth"]):
            current_prefs["verbosity"] = "detailed"
            updated = True

        if any(w in lower for w in ["speak casually", "more casual", "chill tone", "slang"]):
            current_prefs["formality"] = "casual"
            updated = True

        if updated:
            self.db.set_user_attribute(user_id, "style_preferences", current_prefs)
