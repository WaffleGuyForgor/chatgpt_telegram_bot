import re
import logging
from typing import List, Dict, Optional, Tuple

import openai_utils

logger = logging.getLogger(__name__)


class GroupEngine:
    def __init__(self, db):
        self.db = db

    def is_digest_request(self, text: str) -> bool:
        lower = text.lower().strip()
        patterns = [
            r"\bwhat did i miss\b",
            r"\bwhat happened while i was (away|gone|out)\b",
            r"\bsummarize the (recent|last) (messages|chat|discussion)\b",
            r"\bwhat did (we|everyone) decide\b",
            r"\bcatch me up\b",
            r"\bgroup (summary|digest)\b"
        ]
        return any(re.search(p, lower) for p in patterns)

    async def generate_group_digest(self, chat_id: int, requester_name: str = "") -> str:
        """
        Generates a concise group digest from recent messages.
        Includes key decisions, main topics, and open items.
        """
        buffer = self.db.get_group_message_buffer(chat_id)
        if not buffer or len(buffer) < 3:
            return "Not much has happened recently in this chat to summarize!"

        formatted_turns = []
        for item in buffer[-40:]:
            formatted_turns.append(f"[{item.get('time', '')}] {item.get('name', 'Member')}: {item.get('text', '')}")

        history_str = "\n".join(formatted_turns)

        system_prompt = (
            "You are an intelligent group chat assistant. Analyze recent group conversation messages and produce a high-value summary digest.\n"
            "Format the digest into crisp sections:\n"
            "• 📌 Decisions Made\n"
            "• 💬 Key Topics\n"
            "• ❓ Open Questions / Action Items\n\n"
            "Do NOT summarize trivial banter. Be concise, direct, and factual."
        )

        user_prompt = f"Recent Group Conversation:\n{history_str}\n\nPlease provide the digest for {requester_name or 'the group'}."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        digest = await openai_utils.fast_chat_completion(messages, temperature=0.3, max_tokens=600)
        return digest or "Could not generate group digest right now."

    def format_group_message_context(
        self,
        sender_name: str,
        text: str,
        reply_to_sender: Optional[str] = None,
        reply_to_text: Optional[str] = None
    ) -> str:
        """
        Formats a group message with speaker identity and reply context.
        """
        if reply_to_sender and reply_to_text:
            snippet = reply_to_text[:80].replace("\n", " ")
            return f"[{sender_name} replying to {reply_to_sender} (\"{snippet}\")]: {text}"
        return f"[{sender_name}]: {text}"
