import base64
import logging
import asyncio
from io import BytesIO
from typing import List, Dict, Optional, Tuple, AsyncGenerator

import tiktoken
from openai import AsyncOpenAI, BadRequestError, RateLimitError, APIError

import config

logger = logging.getLogger(__name__)

# Initialize unified AsyncOpenAI client
client = AsyncOpenAI(
    api_key=config.llm_api_key,
    base_url=config.llm_base_url,
    timeout=60.0,
    max_retries=2
)

# Optional embedding client
embedding_client = AsyncOpenAI(
    api_key=config.embedding_api_key,
    base_url=config.embedding_base_url,
    timeout=30.0,
    max_retries=2
)

DEFAULT_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1200,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


async def retry_with_backoff(coro_fn, max_retries: int = 3, initial_delay: float = 1.5):
    """Executes an async call with exponential backoff on rate limits or network glitches."""
    delay = initial_delay
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn()
        except RateLimitError as e:
            last_err = e
            logger.warning(f"Rate limit encountered (attempt {attempt}/{max_retries}). Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2
        except APIError as e:
            last_err = e
            if attempt == max_retries or "context_length_exceeded" in str(e).lower():
                raise
            logger.warning(f"API Error (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    if last_err:
        raise last_err


class ChatGPT:
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.default_model
        self._client = client

    async def send_message(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1200
    ) -> Tuple[str, Tuple[int, int]]:
        """Sends a full list of messages directly to the LLM."""
        options = dict(DEFAULT_COMPLETION_OPTIONS)
        options["temperature"] = temperature
        options["max_tokens"] = max_tokens

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options
            )

        r = await retry_with_backoff(_call)
        answer = r.choices[0].message.content or ""
        answer = answer.strip()

        usage = getattr(r, "usage", None)
        n_input_tokens = usage.prompt_tokens if usage else 0
        n_output_tokens = usage.completion_tokens if usage else 0
        return answer, (n_input_tokens, n_output_tokens)

    async def send_message_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1200
    ) -> AsyncGenerator[Tuple[str, str, Tuple[int, int]], None]:
        """Streams message completion tokens."""
        options = dict(DEFAULT_COMPLETION_OPTIONS)
        options["temperature"] = temperature
        options["max_tokens"] = max_tokens
        options["stream"] = True

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options
            )

        r_gen = await retry_with_backoff(_call)
        answer = ""
        n_input_tokens = 0
        n_output_tokens = 0

        async for r_item in r_gen:
            if not r_item.choices:
                continue
            delta = r_item.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                answer += content
                yield "not_finished", answer, (n_input_tokens, n_output_tokens)

        # Estimate tokens at completion
        n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
        yield "finished", answer.strip(), (n_input_tokens, n_output_tokens)

    def _count_tokens_from_messages(self, messages: List[Dict], answer: str, model: str = "") -> Tuple[int, int]:
        try:
            encoding = tiktoken.encoding_for_model(model or self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

        n_input = 0
        for m in messages:
            n_input += 3
            content = m.get("content", "")
            if isinstance(content, list):
                for sub in content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        n_input += len(encoding.encode(sub.get("text", "")))
            else:
                n_input += len(encoding.encode(str(content)))

        n_input += 2
        n_output = 1 + len(encoding.encode(answer))
        return n_input, n_output


async def fast_chat_completion(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800
) -> str:
    """Lightweight non-blocking helper for background tasks (summarization, memory extraction)."""
    target_model = model or config.memory_model
    try:
        r = await client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=40.0
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"fast_chat_completion error with model {target_model}: {e}")
        return ""


async def get_embedding(text: str) -> Optional[List[float]]:
    """Fetches text embedding if an embedding provider is reachable, otherwise returns None for hybrid fallback."""
    if not text or not config.embedding_api_key:
        return None
    try:
        res = await embedding_client.embeddings.create(
            model=config.embedding_model,
            input=text[:2000]
        )
        if res.data:
            return res.data[0].embedding
    except Exception as e:
        logger.debug(f"Embedding generation skipped/failed: {e}")
    return None


async def transcribe_audio(audio_file: BytesIO) -> str:
    """Transcribes audio using Whisper or speech-to-text API."""
    try:
        r = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return r.text or ""
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        raise


async def generate_images(prompt: str, n_images: int = 1, size: str = "1024x1024") -> List[bytes]:
    """Generates images using OpenAI/compatible image API."""
    r = await client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=n_images,
        size=size
    )
    images = [base64.b64decode(item.b64_json) for item in r.data if getattr(item, "b64_json", None)]
    return images
