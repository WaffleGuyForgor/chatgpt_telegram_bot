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
8. NEVER open a response with filler like 'Certainly!', 'Absolutely!', 'Of course!', 'Great question!', 'Sure!', or 'I'd be happy to help'. Start with the substance.
9. Do NOT greet the user at the start of responses — no 'Hey!', 'Hi there!' openers unless the user is greeting you for the first time in a while.
10. Do NOT restate the user's question back to them unless clarification is genuinely needed.
11. Do NOT append a conclusion/summary section to every answer, and do NOT end with 'Let me know if you want...', 'Hope this helps!', or similar closers.
12. Vary your openings, transitions, and structure naturally from answer to answer — never use one fixed template for every response.
13. If you don't know something, say so plainly. Never fabricate facts, memories, file contents, or tool usage.
14. Users write in fragments ('yeah', 'nah', 'the second one', 'do that again', 'why'). Resolve them from the recent conversation and the active object instead of asking them to repeat everything.
15. If the user says you misunderstood ('no, I meant the other file'), correct course immediately: acknowledge briefly at most, do not defend the earlier interpretation, and do not repeat the same mistake.
16. When asked for another version / to regenerate, produce a genuinely different take (different wording, structure, or angle) for the same task — not a near-copy.
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
        user_timezone: Optional[str] = None,
        mood_hint: Optional[str] = None,
        length_hint: Optional[str] = None,
        extra_hint: Optional[str] = None
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

        # Style preferences — scope priority (§13):
        # per-conversation override > global user preference > default
        global_prefs = {}
        if entity_id > 0:
            global_prefs = self.db.get_user_attribute(entity_id, "style_preferences") or {}
        conv_state = self.db.get_conversational_state(entity_id) or {}
        conv_override = conv_state.get("style_override") or {}

        # Merge: conversation-level wins key-by-key over global
        prefs = {**global_prefs, **conv_override}
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
            if prefs.get("emojis") is False:
                style_lines.append("- Do not use emojis in responses.")
            if len(style_lines) > 1:
                parts.append("\n".join(style_lines))

        # Per-turn soft mood signal (ephemeral — never stored, never certain)
        if mood_hint:
            parts.append(mood_hint)

        # Per-turn response length guidance
        if length_hint:
            parts.append(length_hint)

        # Per-turn extra behavioural hint (session mode, repair, regeneration…)
        if extra_hint:
            parts.append(extra_hint)

        return "\n\n".join(parts)

    # Phrases that scope a preference to the current conversation only
    _CONVERSATIONAL_MARKERS = [
        "for this chat", "in this chat", "this conversation", "for now",
        "this time", "just this once", "for this topic", "in this thread",
    ]
    # Phrases that make a preference explicitly global/durable
    _GLOBAL_MARKERS = [
        "from now on", "always", "in general", "i prefer", "i like",
        "remember that i", "every time", "by default",
    ]
    # Reactions to the current answer — conversational scope by default (§15)
    _REACTION_MARKERS = [
        "too long", "too verbose", "too wordy", "too short", "too brief",
        "don't talk like that", "dont talk like that", "stop talking like that",
    ]

    def adapt_user_style_from_message(
        self,
        user_id: int,
        text: str,
        entity_id: Optional[int] = None
    ):
        """
        Learns communication preferences with the correct scope (§13/§15):
        - GLOBAL (user profile) for stated long-term preferences
          ("from now on be concise", "I prefer short answers")
        - CONVERSATIONAL (this chat only) for one-off reactions
          ("too long", "don't talk like that", "for this chat be detailed")
        Momentary frustration is never stored as a durable preference.
        """
        lower = text.lower()

        desired: Dict[str, Any] = {}
        is_reaction = any(m in lower for m in self._REACTION_MARKERS)

        if any(w in lower for w in ["be concise", "keep it short", "shorter", "too long", "too verbose", "brief answer", "bullet points only", "too wordy"]):
            desired["verbosity"] = "low"
        elif any(w in lower for w in ["too short", "too brief", "explain in detail", "elaborate", "give more detail", "in depth", "more detail", "be detailed", "more detailed", "detailed answers"]):
            desired["verbosity"] = "detailed"

        if any(w in lower for w in ["speak casually", "more casual", "chill tone", "slang", "don't talk like that", "dont talk like that", "stop being formal", "talk normally"]):
            desired["formality"] = "casual"

        if any(w in lower for w in ["stop using emojis", "no emojis", "don't use emojis", "dont use emojis", "without emojis"]):
            desired["emojis"] = False
        elif "use emojis" in lower or "more emojis" in lower:
            desired["emojis"] = True

        if not desired:
            return

        conversational_scope = (
            any(m in lower for m in self._CONVERSATIONAL_MARKERS)
            or (is_reaction and not any(m in lower for m in self._GLOBAL_MARKERS))
        )

        if conversational_scope:
            target = entity_id if entity_id is not None else user_id
            state = self.db.get_conversational_state(target) or {}
            state = {k: v for k, v in state.items() if k != "_id"}  # never $set immutable _id
            override = state.get("style_override") or {}
            override.update(desired)
            state["style_override"] = override
            self.db.set_conversational_state(target, state)
        elif user_id > 0:
            current_prefs = self.db.get_user_attribute(user_id, "style_preferences") or {
                "verbosity": "balanced",
                "formality": "natural",
                "technical_detail": "adaptive"
            }
            current_prefs.update(desired)
            self.db.set_user_attribute(user_id, "style_preferences", current_prefs)

    def explain_response_style(self, user_id: int, entity_id: int) -> Optional[str]:
        """
        Natural explanation for 'why did you answer that way?' (§14).
        Surfaces active preferences without exposing internal mechanics.
        """
        reasons = []
        conv_state = self.db.get_conversational_state(entity_id) or {}
        override = conv_state.get("style_override") or {}
        prefs = (self.db.get_user_attribute(user_id, "style_preferences") or {}) if user_id > 0 else {}

        if override.get("verbosity") == "low":
            reasons.append("you asked me to keep it brief in this chat")
        elif prefs.get("verbosity") == "low":
            reasons.append("you usually prefer concise answers, so I kept it brief")
        if override.get("verbosity") == "detailed":
            reasons.append("you asked for detailed answers in this chat")
        elif prefs.get("verbosity") == "detailed":
            reasons.append("you usually prefer detailed explanations")
        if prefs.get("formality") == "casual" or override.get("formality") == "casual":
            reasons.append("you prefer a casual tone")
        if prefs.get("emojis") is False or override.get("emojis") is False:
            reasons.append("you asked me not to use emojis")

        if not reasons:
            return None
        if len(reasons) == 1:
            return reasons[0][0].upper() + reasons[0][1:] + "."
        return "A few things shaped that: " + "; ".join(reasons) + "."
