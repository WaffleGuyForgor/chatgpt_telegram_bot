import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import json
import logging
from typing import List, Dict, Optional, Any, Tuple

import config
import openai_utils
from personality import PersonalityEngine
from memory import MemoryEngine
from group_engine import GroupEngine

logger = logging.getLogger(__name__)


class ConversationContextEngine:
    def __init__(self, db):
        self.db = db
        self.personality = PersonalityEngine(db)
        self.memory = MemoryEngine(db)
        self.group = GroupEngine(db)

    async def build_llm_messages(
        self,
        entity_id: int,
        current_user_text: str,
        user_name: str = "",
        is_group: bool = False,
        group_title: str = "",
        image_data_url: Optional[str] = None,
        document_context: Optional[str] = None,
        reply_context: Optional[str] = None,
        mood_hint: Optional[str] = None,
        length_hint: Optional[str] = None
    ) -> List[Dict]:
        """
        Assembles full context with stable priority hierarchy:
        1. Base identity & behavioral rules
        2. Time & user style preferences
        3. Relevant retrieved long-term memories
        4. Rolling conversation summary
        5. Active task / conversational state
        6. Recent conversation messages
        7. Current message (multimodal if image/document attached)
        """
        messages = []

        # 1. System Prompt
        user_tz = self.db.get_user_attribute(entity_id, "timezone") if entity_id > 0 else None
        system_prompt = self.personality.get_system_prompt(
            entity_id=entity_id,
            user_name=user_name,
            is_group=is_group,
            group_title=group_title,
            user_timezone=user_tz,
            mood_hint=mood_hint,
            length_hint=length_hint
        )

        # 2. Retrieve Relevant Long-Term Memory
        # Strict isolation: user memory for private chats, group memory for groups!
        scope = f"group:{entity_id}" if is_group else f"user:{entity_id}"
        relevant_memories = await self.memory.retrieve_relevant_memories(
            scope=scope,
            entity_id=entity_id,
            query_text=current_user_text,
            limit=config.memory_retrieval_limit
        )
        memory_str = self.memory.format_memory_for_prompt(relevant_memories)

        # 3. Rolling Summary of Older Dialog Context
        rolling_summary = self.db.get_dialog_summary(entity_id)

        # Combine System instructions
        system_full = system_prompt
        if memory_str:
            system_full += f"\n\n{memory_str}"
        if rolling_summary:
            system_full += f"\n\n<conversation_summary>\nSummary of previous conversation:\n{rolling_summary}\n</conversation_summary>"

        # 4. Active Object / Follow-up State
        state = self.db.get_conversational_state(entity_id)
        last_output = state.get("last_output")
        if last_output and any(p in current_user_text.lower() for p in ["it", "this", "that", "shorter", "casual", "alternative", "another version", "rewrite", "more"]):
            system_full += f"\n\n<active_reference_object>\nThe user may be referring to the previous output:\n\"\"\"\n{last_output[:800]}\n\"\"\"\n</active_reference_object>"

        messages.append({"role": "system", "content": system_full})

        # 5. Recent Dialog History
        dialog_messages = self.db.get_dialog_messages(entity_id)
        # Only take last 10 messages to keep prompt clean
        recent_turns = dialog_messages[-10:] if len(dialog_messages) > 10 else dialog_messages
        for turn in recent_turns:
            user_val = turn.get("user", "")
            bot_val = turn.get("bot", "")
            if isinstance(user_val, str):
                messages.append({"role": "user", "content": user_val})
            elif isinstance(user_val, list):
                messages.append({"role": "user", "content": user_val})
            if bot_val:
                messages.append({"role": "assistant", "content": bot_val})

        # 6. Current User Turn (with Multimodal context if present)
        current_payload = current_user_text
        if reply_context:
            current_payload = f"[Replying to: \"{reply_context}\"]\n\n{current_payload}"
        if document_context:
            current_payload = f"{document_context}\n\n{current_payload}"

        if image_data_url:
            user_content = [
                {"type": "text", "text": current_payload or "Please inspect this image."},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}}
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": current_payload})

        return messages

    async def maybe_summarize_in_background(self, entity_id: int):
        """
        Background task: checks if dialog history exceeds threshold.
        If so, summarizes older turns and updates the rolling summary.
        """
        dialog_messages = self.db.get_dialog_messages(entity_id)
        if len(dialog_messages) < config.rolling_summary_threshold:
            return

        # Split: summarize older turns, keep the most recent 8
        older_turns = dialog_messages[:-8]
        recent_turns = dialog_messages[-8:]

        history_text = []
        for t in older_turns:
            u = t.get("user", "")
            b = t.get("bot", "")
            if isinstance(u, str):
                history_text.append(f"User: {u}")
            history_text.append(f"Assistant: {b[:200]}")

        existing_summary = self.db.get_dialog_summary(entity_id)
        prompt = (
            "Summarize this conversation concisely into 2-4 key bullet points, preserving key facts, user goals, and decisions.\n"
            f"Previous summary: {existing_summary}\n\n"
            f"Conversation to incorporate:\n" + "\n".join(history_text)
        )

        messages = [
            {"role": "system", "content": "You are a concise conversation summarizer."},
            {"role": "user", "content": prompt}
        ]

        new_summary = await openai_utils.fast_chat_completion(messages, temperature=0.2, max_tokens=350)
        if new_summary:
            self.db.set_dialog_summary(entity_id, new_summary)
            self.db.set_dialog_messages(entity_id, recent_turns)
            logger.info(f"Updated rolling summary for entity {entity_id}")
