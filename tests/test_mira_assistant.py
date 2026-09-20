import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add project root and bot directory to sys.path
root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "bot"))

from bot.memory import MemoryEngine, cosine_similarity, tokenize
from bot.personality import PersonalityEngine
from bot.group_engine import GroupEngine
from bot.bot import split_text_into_chunks


class DummyDatabase:
    """In-memory mock for Database to test memory and context engines without a live Mongo instance."""
    def __init__(self):
        self.memories = []
        self.users = {}
        self.chats = {}
        self.states = {}

    def get_active_memories(self, scope, entity_id):
        return [m for m in self.memories if m.get("scope") == scope and m.get("entity_id") == entity_id and m.get("is_active", True)]

    def add_memory(self, scope, entity_id, content, category="personal", importance=0.5, keywords=None, embedding=None):
        mem_id = f"mem_{len(self.memories) + 1}"
        self.memories.append({
            "_id": mem_id,
            "scope": scope,
            "entity_id": entity_id,
            "content": content,
            "category": category,
            "importance": importance,
            "keywords": keywords or [],
            "embedding": embedding,
            "updated_at": datetime.now(),
            "is_active": True
        })
        return mem_id

    def delete_memories_by_keyword(self, scope, entity_id, keyword):
        count = 0
        for m in self.memories:
            if m.get("scope") == scope and m.get("entity_id") == entity_id and keyword.lower() in m.get("content", "").lower():
                m["is_active"] = False
                count += 1
        return count

    def touch_memory_recalled(self, memory_ids):
        pass

    def get_user_attribute(self, user_id, key):
        return self.users.get(user_id, {}).get(key)

    def set_user_attribute(self, user_id, key, value):
        if user_id not in self.users:
            self.users[user_id] = {}
        self.users[user_id][key] = value

    def get_conversational_state(self, entity_id):
        return self.states.get(entity_id, {})

    def set_conversational_state(self, entity_id, data):
        self.states[entity_id] = data

    def get_group_message_buffer(self, chat_id):
        return [
            {"time": "12:00", "name": "Alice", "text": "Let's release Project Hermes on Saturday."},
            {"time": "12:01", "name": "Bob", "text": "Agreed, Saturday is better."},
            {"time": "12:02", "name": "Alice", "text": "Saturday confirmed."}
        ]


class TestMiraAssistant(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = DummyDatabase()
        self.memory_engine = MemoryEngine(self.db)
        self.personality = PersonalityEngine(self.db)
        self.group = GroupEngine(self.db)

    def test_natural_language_memory_intents(self):
        # Explicit remember
        intent, arg = self.memory_engine.detect_memory_intent("Remember that my server is called Atlas")
        self.assertEqual(intent, "explicit_remember")
        self.assertEqual(arg, "my server is called Atlas")

        # Forget
        intent, arg = self.memory_engine.detect_memory_intent("Forget my server name")
        self.assertEqual(intent, "forget")
        self.assertEqual(arg, "my server name")

        # Show memories
        intent, arg = self.memory_engine.detect_memory_intent("What do you remember about me?")
        self.assertEqual(intent, "show")

        # Off the record
        intent, arg = self.memory_engine.detect_memory_intent("Don't remember this conversation")
        self.assertEqual(intent, "off_the_record")

    async def test_memory_scoring_and_retrieval(self):
        scope = "user:123"
        entity_id = 123
        self.db.add_memory(scope, entity_id, "User is developing a bot named Hermes", "project", importance=0.9, keywords=["hermes", "bot"])
        self.db.add_memory(scope, entity_id, "User likes pizza", "personal", importance=0.3, keywords=["pizza"])

        retrieved = await self.memory_engine.retrieve_relevant_memories(scope, entity_id, "How is project Hermes going?", limit=3)
        self.assertTrue(len(retrieved) > 0)
        self.assertIn("Hermes", retrieved[0]["content"])

    def test_user_style_adaptation(self):
        user_id = 999
        self.personality.adapt_user_style_from_message(user_id, "Please keep it short and be concise")
        prefs = self.db.get_user_attribute(user_id, "style_preferences")
        self.assertEqual(prefs.get("verbosity"), "low")

    def test_group_digest_detection(self):
        self.assertTrue(self.group.is_digest_request("What did I miss?"))
        self.assertTrue(self.group.is_digest_request("What did everyone decide?"))
        self.assertFalse(self.group.is_digest_request("Hello bot, how are you?"))

    def test_text_splitting(self):
        long_text = "Paragraph one.\n\n" + ("A" * 3000) + "\n\nParagraph three.\n\n" + ("B" * 2000)
        chunks = split_text_into_chunks(long_text, chunk_size=3500)
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 3600)


if __name__ == "__main__":
    unittest.main()
