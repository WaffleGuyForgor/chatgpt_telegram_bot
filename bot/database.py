from typing import Optional, Any, List, Dict
import logging
import pymongo
import uuid
from datetime import datetime

import config

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        logger.info(f"Connecting to MongoDB: {config.mongodb_uri[:50]}...")
        self.client = pymongo.MongoClient(
            config.mongodb_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
        )
        # Verify connection immediately
        try:
            self.client.admin.command("ping")
            logger.info("MongoDB connection successful!")
        except Exception as e:
            logger.error(f"MongoDB connection FAILED: {e}")
            logger.error("The bot will start but database features will not work until MongoDB is reachable.")

        self.db = self.client["chatgpt_telegram_bot"]

        self.user_collection = self.db["user"]
        self.dialog_collection = self.db["dialog"]
        self.chat_collection = self.db["chat"]
        self.settings_collection = self.db["settings"]
        self.memories_collection = self.db["memories"]
        self.state_collection = self.db["conversational_state"]

        # Ensure indexes for fast, non-blocking queries
        self._ensure_indexes()

    def _ensure_indexes(self):
        try:
            self.memories_collection.create_index([("scope", pymongo.ASCENDING), ("is_active", pymongo.ASCENDING)])
            self.memories_collection.create_index([("entity_id", pymongo.ASCENDING), ("category", pymongo.ASCENDING)])
            self.memories_collection.create_index([("updated_at", pymongo.DESCENDING)])
            self.dialog_collection.create_index([("user_id", pymongo.ASCENDING)])
            self.chat_collection.create_index([("last_active", pymongo.DESCENDING)])
        except Exception:
            pass

    # ---------------- Global Settings ----------------
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        doc = self.settings_collection.find_one({"_id": key})
        if doc is not None and "value" in doc:
            return doc["value"]
        return default

    def set_global_setting(self, key: str, value: Any):
        self.settings_collection.update_one(
            {"_id": key},
            {"$set": {"value": value, "updated_at": datetime.now()}},
            upsert=True
        )

    # ---------------- Group / Chat Management ----------------
    def add_or_update_chat(self, chat_id: int, title: str, chat_type: str, username: Optional[str] = None):
        existing = self.chat_collection.find_one({"_id": chat_id})
        if not existing:
            chat_doc = {
                "_id": chat_id,
                "title": title or "Group",
                "type": chat_type,
                "username": username,
                "created_at": datetime.now(),
                "last_active": datetime.now(),
                "auto_reply": False,
                "current_dialog_id": None,
                "current_chat_mode": "assistant",
                "current_model": config.default_model,
                "n_used_tokens": {},
                "recent_buffer": []  # rolling buffer for digests
            }
            self.chat_collection.insert_one(chat_doc)
        else:
            update_fields = {
                "title": title or existing.get("title", "Group"),
                "type": chat_type,
                "last_active": datetime.now()
            }
            if username:
                update_fields["username"] = username
            self.chat_collection.update_one({"_id": chat_id}, {"$set": update_fields})

    def append_group_message_buffer(self, chat_id: int, sender_name: str, sender_id: int, text: str):
        """Appends a message to the group's rolling buffer for digest generation ('What did I miss?')."""
        item = {
            "name": sender_name,
            "sender_id": sender_id,
            "text": text[:500],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        # Keep last 50 group messages in the digest buffer
        self.chat_collection.update_one(
            {"_id": chat_id},
            {
                "$push": {"recent_buffer": {"$each": [item], "$slice": -50}},
                "$set": {"last_active": datetime.now()}
            }
        )

    def get_group_message_buffer(self, chat_id: int) -> List[Dict]:
        doc = self.chat_collection.find_one({"_id": chat_id})
        if doc and "recent_buffer" in doc:
            return doc["recent_buffer"]
        return []

    def get_chat(self, chat_id: int) -> Optional[Dict]:
        return self.chat_collection.find_one({"_id": chat_id})

    def set_chat_attribute(self, chat_id: int, key: str, value: Any):
        self.chat_collection.update_one({"_id": chat_id}, {"$set": {key: value}}, upsert=True)

    def get_chat_attribute(self, chat_id: int, key: str, default: Any = None) -> Any:
        doc = self.chat_collection.find_one({"_id": chat_id})
        if doc and key in doc:
            return doc[key]
        return default

    def get_all_chats(self) -> List[Dict]:
        return list(self.chat_collection.find({}))

    def get_total_groups_count(self) -> int:
        return self.chat_collection.count_documents({"_id": {"$lt": 0}})

    # ---------------- User Management ----------------
    def check_if_user_exists(self, user_id: int, raise_exception: bool = False) -> bool:
        try:
            if self.user_collection.count_documents({"_id": user_id}) > 0:
                return True
            else:
                if raise_exception:
                    raise ValueError(f"User {user_id} does not exist")
                return False
        except Exception as e:
            logger.error(f"MongoDB query failed in check_if_user_exists: {e}")
            if raise_exception:
                raise
            return False

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        user_dict = {
            "_id": user_id,
            "chat_id": chat_id,
            "username": username or "",
            "first_name": first_name or "",
            "last_name": last_name or "",
            "last_interaction": datetime.now(),
            "first_seen": datetime.now(),
            "current_dialog_id": None,
            "current_chat_mode": "assistant",
            "current_model": config.default_model,
            "n_used_tokens": {},
            "timezone": None,
            "language": None,
            "memory_active": True,
            "style_preferences": {
                "verbosity": "balanced",
                "formality": "natural",
                "technical_detail": "adaptive"
            }
        }
        if not self.check_if_user_exists(user_id):
            self.user_collection.insert_one(user_dict)

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        user_dict = self.user_collection.find_one({"_id": user_id})
        return user_dict.get(key, None)

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.user_collection.update_one({"_id": user_id}, {"$set": {key: value}})

    # ---------------- Dialog & Rolling Summary Management ----------------
    def start_new_dialog(self, entity_id: int) -> str:
        """Starts a new dialog for a user (entity_id > 0) or a group (entity_id < 0)."""
        dialog_id = str(uuid.uuid4())

        if entity_id > 0:
            self.check_if_user_exists(entity_id, raise_exception=True)
            chat_mode = self.get_user_attribute(entity_id, "current_chat_mode") or "assistant"
            model = self.get_user_attribute(entity_id, "current_model") or config.default_model
        else:
            chat_mode = self.get_chat_attribute(entity_id, "current_chat_mode", "assistant")
            model = self.get_chat_attribute(entity_id, "current_model", config.default_model)

        dialog_dict = {
            "_id": dialog_id,
            "user_id": entity_id,
            "chat_mode": chat_mode,
            "start_time": datetime.now(),
            "model": model,
            "messages": [],
            "summary": ""
        }
        self.dialog_collection.insert_one(dialog_dict)

        if entity_id > 0:
            self.user_collection.update_one({"_id": entity_id}, {"$set": {"current_dialog_id": dialog_id}})
        else:
            self.set_chat_attribute(entity_id, "current_dialog_id", dialog_id)

        return dialog_id

    def get_dialog_messages(self, entity_id: int, dialog_id: Optional[str] = None) -> List[Dict]:
        if dialog_id is None:
            if entity_id > 0:
                dialog_id = self.get_user_attribute(entity_id, "current_dialog_id")
            else:
                dialog_id = self.get_chat_attribute(entity_id, "current_dialog_id")

        if dialog_id is None:
            dialog_id = self.start_new_dialog(entity_id)

        dialog_dict = self.dialog_collection.find_one({"_id": dialog_id, "user_id": entity_id})
        if dialog_dict is None:
            return []
        return dialog_dict.get("messages", [])

    def set_dialog_messages(self, entity_id: int, dialog_messages: list, dialog_id: Optional[str] = None):
        if dialog_id is None:
            if entity_id > 0:
                dialog_id = self.get_user_attribute(entity_id, "current_dialog_id")
            else:
                dialog_id = self.get_chat_attribute(entity_id, "current_dialog_id")

        if dialog_id is None:
            dialog_id = self.start_new_dialog(entity_id)

        max_msgs = getattr(config, "max_dialog_messages", 30)
        if len(dialog_messages) > max_msgs:
            dialog_messages = dialog_messages[-max_msgs:]

        self.dialog_collection.update_one(
            {"_id": dialog_id, "user_id": entity_id},
            {"$set": {"messages": dialog_messages}}
        )

    def get_dialog_summary(self, entity_id: int, dialog_id: Optional[str] = None) -> str:
        if dialog_id is None:
            if entity_id > 0:
                dialog_id = self.get_user_attribute(entity_id, "current_dialog_id")
            else:
                dialog_id = self.get_chat_attribute(entity_id, "current_dialog_id")
        if not dialog_id:
            return ""
        doc = self.dialog_collection.find_one({"_id": dialog_id, "user_id": entity_id})
        return doc.get("summary", "") if doc else ""

    def set_dialog_summary(self, entity_id: int, summary: str, dialog_id: Optional[str] = None):
        if dialog_id is None:
            if entity_id > 0:
                dialog_id = self.get_user_attribute(entity_id, "current_dialog_id")
            else:
                dialog_id = self.get_chat_attribute(entity_id, "current_dialog_id")
        if dialog_id:
            self.dialog_collection.update_one(
                {"_id": dialog_id, "user_id": entity_id},
                {"$set": {"summary": summary}}
            )

    def update_n_used_tokens(self, entity_id: int, model: str, n_input_tokens: int, n_output_tokens: int):
        if entity_id > 0:
            n_used_tokens_dict = self.get_user_attribute(entity_id, "n_used_tokens") or {}
        else:
            n_used_tokens_dict = self.get_chat_attribute(entity_id, "n_used_tokens", {})

        if model in n_used_tokens_dict:
            n_used_tokens_dict[model]["n_input_tokens"] += n_input_tokens
            n_used_tokens_dict[model]["n_output_tokens"] += n_output_tokens
        else:
            n_used_tokens_dict[model] = {
                "n_input_tokens": n_input_tokens,
                "n_output_tokens": n_output_tokens
            }

        if entity_id > 0:
            self.set_user_attribute(entity_id, "n_used_tokens", n_used_tokens_dict)
        else:
            self.set_chat_attribute(entity_id, "n_used_tokens", n_used_tokens_dict)

    # ---------------- Structured Long-Term Memory ----------------
    def add_memory(
        self,
        scope: str,
        entity_id: int,
        content: str,
        category: str = "personal",
        importance: float = 0.5,
        confidence: float = 0.9,
        keywords: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        mem_id = str(uuid.uuid4())
        doc = {
            "_id": mem_id,
            "scope": scope,
            "entity_id": entity_id,
            "content": content.strip(),
            "category": category,
            "importance": float(importance),
            "confidence": float(confidence),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_recalled_at": datetime.now(),
            "is_active": True,
            "keywords": keywords or [],
            "embedding": embedding
        }
        self.memories_collection.insert_one(doc)
        return mem_id

    def get_active_memories(self, scope: str, entity_id: int) -> List[Dict]:
        return list(self.memories_collection.find({
            "scope": scope,
            "entity_id": entity_id,
            "is_active": True
        }))

    def update_memory(self, memory_id: str, new_content: str, importance: Optional[float] = None):
        fields = {"content": new_content.strip(), "updated_at": datetime.now()}
        if importance is not None:
            fields["importance"] = float(importance)
        self.memories_collection.update_one({"_id": memory_id}, {"$set": fields})

    def delete_memory(self, memory_id: str):
        """Soft delete a memory record."""
        self.memories_collection.update_one({"_id": memory_id}, {"$set": {"is_active": False, "deleted_at": datetime.now()}})

    def delete_memories_by_keyword(self, scope: str, entity_id: int, keyword: str) -> int:
        """Soft delete memories matching a specific topic or keyword."""
        res = self.memories_collection.update_many(
            {
                "scope": scope,
                "entity_id": entity_id,
                "is_active": True,
                "$or": [
                    {"content": {"$regex": keyword, "$options": "i"}},
                    {"keywords": {"$regex": keyword, "$options": "i"}}
                ]
            },
            {"$set": {"is_active": False, "deleted_at": datetime.now()}}
        )
        return res.modified_count

    def clear_all_memories(self, scope: str, entity_id: int) -> int:
        res = self.memories_collection.update_many(
            {"scope": scope, "entity_id": entity_id, "is_active": True},
            {"$set": {"is_active": False, "deleted_at": datetime.now()}}
        )
        return res.modified_count

    def touch_memory_recalled(self, memory_ids: List[str]):
        if memory_ids:
            self.memories_collection.update_many(
                {"_id": {"$in": memory_ids}},
                {"$set": {"last_recalled_at": datetime.now()}}
            )

    # ---------------- Conversational Task State (Short-lived) ----------------
    def set_conversational_state(self, entity_id: int, state_data: Dict):
        state_data["updated_at"] = datetime.now()
        self.state_collection.update_one({"_id": entity_id}, {"$set": state_data}, upsert=True)

    def get_conversational_state(self, entity_id: int) -> Dict:
        doc = self.state_collection.find_one({"_id": entity_id})
        return doc or {}

    # ---------------- Stats Helpers for Owner Panel ----------------
    def get_all_user_ids(self) -> List[int]:
        return [doc["_id"] for doc in self.user_collection.find({}, {"_id": 1})]

    def get_total_users_count(self) -> int:
        return self.user_collection.count_documents({})

    def get_active_chats_count(self) -> int:
        return self.dialog_collection.count_documents({})

    def get_total_memories_count(self) -> int:
        return self.memories_collection.count_documents({"is_active": True})
