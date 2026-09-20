import io
import os
import base64
import logging
from typing import Optional, Tuple, Dict, Any

from telegram import Update
from telegram.ext import CallbackContext

import openai_utils

logger = logging.getLogger(__name__)

# Supported text extensions for direct reading
TEXT_EXTENSIONS = {
    ".txt", ".csv", ".json", ".md", ".py", ".js", ".ts", ".html",
    ".css", ".yml", ".yaml", ".env", ".sh", ".sql", ".xml", ".log"
}


class MultimodalInterpreter:
    def __init__(self, db):
        self.db = db

    async def process_voice(self, update: Update, context: CallbackContext) -> Tuple[str, float]:
        """
        Downloads a voice message, transcribes it via Whisper, and returns (transcript, duration).
        """
        voice = update.message.voice
        if not voice:
            return "", 0.0

        voice_file = await context.bot.get_file(voice.file_id)
        buf = io.BytesIO()
        await voice_file.download_to_memory(buf)
        buf.name = "voice.oga"
        buf.seek(0)

        transcript = await openai_utils.transcribe_audio(buf)
        return transcript.strip(), float(voice.duration)

    async def process_photo(
        self,
        update: Update,
        context: CallbackContext,
        entity_id: int
    ) -> Tuple[str, Optional[str]]:
        """
        Downloads the highest resolution photo and returns (caption, base64_image_data_url).
        Saves the image into conversational state for follow-up questions.
        """
        photos = update.message.photo
        if not photos:
            return update.message.caption or "", None

        photo = photos[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await photo_file.download_to_memory(buf)
        buf.seek(0)

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"
        caption = update.message.caption or ""

        # Store in conversational state for natural follow-up ("make it darker", "explain this part")
        self.db.set_conversational_state(entity_id, {
            "last_image_data_url": data_url,
            "last_image_caption": caption
        })

        return caption, data_url

    async def process_document(
        self,
        update: Update,
        context: CallbackContext,
        max_chars: int = 15000
    ) -> Tuple[str, str]:
        """
        Downloads and extracts text from supported documents (PDF, TXT, CSV, JSON, Code).
        Returns (caption, document_text).
        """
        doc = update.message.document
        if not doc:
            return update.message.caption or "", ""

        caption = update.message.caption or ""
        file_name = doc.file_name or "document"
        _, ext = os.path.splitext(file_name.lower())

        file_obj = await context.bot.get_file(doc.file_id)
        buf = io.BytesIO()
        await file_obj.download_to_memory(buf)
        buf.seek(0)

        extracted_text = ""

        # 1. PDF handling
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(buf)
                pages_text = []
                for i, page in enumerate(reader.pages[:25]):  # Read up to first 25 pages
                    t = page.extract_text()
                    if t:
                        pages_text.append(f"--- Page {i+1} ---\n{t.strip()}")
                extracted_text = "\n\n".join(pages_text)
            except Exception as e:
                logger.warning(f"PDF extraction error: {e}")
                extracted_text = f"[Could not extract text from PDF: {e}]"

        # 2. Text / Code / CSV / JSON handling
        elif ext in TEXT_EXTENSIONS or doc.mime_type and ("text" in doc.mime_type or "json" in doc.mime_type):
            try:
                raw_bytes = buf.getvalue()
                try:
                    extracted_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    extracted_text = raw_bytes.decode("latin-1", errors="replace")
            except Exception as e:
                logger.warning(f"Text file extraction error: {e}")
                extracted_text = f"[Error reading text file: {e}]"
        else:
            extracted_text = f"[Attached binary file: {file_name} ({doc.mime_type})]"

        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars] + f"\n\n[... content truncated to first {max_chars} characters ...]"

        formatted_doc = f"[Document: {file_name}]\n{extracted_text}"
        return caption, formatted_doc

    def resolve_conversational_image(self, entity_id: int, user_text: str) -> Optional[str]:
        """
        If the user is asking a follow-up about an image previously sent in the conversation
        (e.g. 'what about this image?', 'make it darker', 'look at the top right'),
        re-attaches the active image from conversational state.
        """
        state = self.db.get_conversational_state(entity_id)
        last_image = state.get("last_image_data_url")
        if not last_image:
            return None

        # Check for image follow-up pronouns/references
        lower = user_text.lower()
        triggers = ["the image", "this picture", "the screenshot", "the photo", "make it", "what is this", "in this", "look closer"]
        if any(t in lower for t in triggers):
            return last_image

        return None
