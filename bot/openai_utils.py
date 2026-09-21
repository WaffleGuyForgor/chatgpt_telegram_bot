import base64
import logging
import asyncio
import itertools
import time
from io import BytesIO
from typing import List, Dict, Optional, Tuple, AsyncGenerator

import tiktoken
from openai import (
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
    APIError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
)

import config

logger = logging.getLogger(__name__)


class ProviderNotConfiguredError(Exception):
    """Raised when a model's provider has no API keys configured."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"No API keys configured for provider '{provider}'")


class APIKeyPool:
    """
    Round-robin API key pool with cooldown.
    When a key hits a rate limit, it's temporarily excluded from rotation.
    """

    def __init__(self, keys: List[str], base_url: str, cooldown_seconds: int = 60):
        self._keys = keys
        self._base_url = base_url
        self._cooldown = cooldown_seconds
        self._cycle = itertools.cycle(range(len(keys)))
        self._clients: List[AsyncOpenAI] = []
        self._cooldowns: Dict[int, float] = {}  # index -> cooldown_until timestamp

        for key in keys:
            self._clients.append(
                AsyncOpenAI(api_key=key, base_url=base_url, timeout=60.0, max_retries=0)
            )

        logger.info(f"API key pool initialized: {len(keys)} keys for {base_url}")

    @property
    def num_keys(self) -> int:
        return len(self._keys)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_next_client(self) -> Tuple[AsyncOpenAI, int]:
        """Returns the next available client via round-robin, skipping keys in cooldown."""
        if not self._keys:
            raise ProviderNotConfiguredError(self._base_url)
        now = time.time()
        attempts = 0
        while attempts < len(self._keys):
            idx = next(self._cycle)
            cooldown_until = self._cooldowns.get(idx, 0)
            if now >= cooldown_until:
                return self._clients[idx], idx
            attempts += 1

        # All keys in cooldown — return the one whose cooldown expires soonest
        soonest_idx = min(self._cooldowns, key=self._cooldowns.get)
        return self._clients[soonest_idx], soonest_idx

    def mark_rate_limited(self, idx: int):
        """Marks a key as rate-limited, putting it in cooldown."""
        self._cooldowns[idx] = time.time() + self._cooldown
        logger.warning(f"API key #{idx} rate-limited, cooling down for {self._cooldown}s")

    def get_stats(self) -> Dict:
        now = time.time()
        active = sum(1 for i in range(len(self._keys)) if now >= self._cooldowns.get(i, 0))
        return {
            "total_keys": len(self._keys),
            "active_keys": active,
            "cooldown_keys": len(self._keys) - active,
        }


# Build per-provider API key pools (Groq = default, OpenRouter, Dahl)
DEFAULT_PROVIDER = "groq"

_provider_pools: Dict[str, APIKeyPool] = {}
for _provider_name, _provider_cfg in config.provider_registry.items():
    _keys = _provider_cfg.get("keys") or []
    _base = _provider_cfg.get("base_url") or ""
    if _keys:
        _provider_pools[_provider_name] = APIKeyPool(_keys, _base, cooldown_seconds=60)
    else:
        logger.warning(f"Provider '{_provider_name}' has no API keys — its models will be unavailable.")

# Backward-compatible handle on the default (Groq) pool — used for Whisper/images.
# Falls back to any configured pool so audio/images keep working if only e.g. OpenRouter is set up.
_api_pool: Optional[APIKeyPool] = _provider_pools.get(DEFAULT_PROVIDER) or next(
    iter(_provider_pools.values()), None
)

# Keep legacy module-level names for any external imports
_llm_keys = config.llm_api_keys
_llm_base_url = config.llm_base_url


def get_provider_for_model(model: str) -> str:
    """Resolves which provider serves a given model (from models.yml `provider:` field)."""
    info = config.models.get("info", {}).get(model, {})
    return info.get("provider", DEFAULT_PROVIDER)


def get_pool_for_model(model: str) -> APIKeyPool:
    """Returns the API key pool that serves the given model."""
    provider = get_provider_for_model(model)
    pool = _provider_pools.get(provider)
    if pool is None:
        raise ProviderNotConfiguredError(provider)
    return pool


def provider_is_configured(provider: str) -> bool:
    """True if the provider has at least one API key configured."""
    return provider in _provider_pools


def get_provider_stats() -> Dict[str, Dict]:
    """Per-provider key pool stats for /status and the owner panel."""
    stats = {}
    for name, pool in _provider_pools.items():
        s = pool.get_stats()
        s["base_url"] = pool.base_url
        stats[name] = s
    return stats


# ---------------- Model performance stats (§28) ----------------
_model_stats: Dict[str, Dict] = {}


def _record_model_call(model: str, latency: float, ok: bool, rate_limited: bool = False, error: Optional[Exception] = None):
    s = _model_stats.setdefault(model, {
        "calls": 0, "errors": 0, "rate_limits": 0, "total_latency": 0.0, "last_error": None
    })
    s["calls"] += 1
    s["total_latency"] += latency
    if rate_limited:
        s["rate_limits"] += 1
    if not ok:
        s["errors"] += 1
        s["last_error"] = str(error)[:200] if error else None


def get_model_stats() -> Dict[str, Dict]:
    """Rolling per-model performance: calls, errors, rate limits, average latency."""
    out = {}
    for model, s in _model_stats.items():
        out[model] = {
            **s,
            "avg_latency": round(s["total_latency"] / s["calls"], 2) if s["calls"] else 0.0,
        }
    return out

# Embedding client (single key)
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


def get_api_pool() -> APIKeyPool:
    """Returns the default (Groq) API key pool for stats display."""
    return _api_pool


class ChatGPT:
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.default_model
        self.provider = get_provider_for_model(self.model)
        self.used_model = self.model      # actual model that answered (differs if fallback engaged)
        self.used_fallback = False

    def _candidate_models(self) -> List[str]:
        """Primary model + configured fallback chain (§27), deduplicated."""
        chain = [self.model]
        for m in config.fallback_models:
            if m != self.model and m not in chain:
                chain.append(m)
        return chain

    async def send_message(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1200
    ) -> Tuple[str, Tuple[int, int]]:
        """Sends messages to the LLM with key rotation + optional fallback chain (§27).

        Tries the primary model first; if its provider pool is exhausted and
        config.fallback_models is set, walks the chain (each with its own
        provider pool). Sets self.used_fallback / self.used_model accordingly.
        """
        options = dict(DEFAULT_COMPLETION_OPTIONS)
        options["temperature"] = temperature
        options["max_tokens"] = max_tokens

        last_error = None
        for model in self._candidate_models():
            try:
                answer, tokens = await self._send_with_model(model, messages, options)
                if model != self.model:
                    self.used_fallback = True
                    self.used_model = model
                    logger.warning(f"Fallback engaged: '{self.model}' -> '{model}'")
                return answer, tokens
            except BadRequestError:
                raise  # request-shaped problem (e.g. context length) — other models won't help
            except Exception as e:
                last_error = e
                logger.warning(f"Model '{model}' failed: {e}. Trying next in chain…")
                continue
        raise last_error or Exception("All models in the fallback chain are unavailable")

    async def _send_with_model(
        self,
        model: str,
        messages: List[Dict],
        options: Dict
    ) -> Tuple[str, Tuple[int, int]]:
        """Single-model completion with key rotation across that model's provider pool."""
        pool = get_pool_for_model(model)
        last_error = None
        for attempt in range(pool.num_keys + 1):
            client, key_idx = pool._get_next_client()
            start = time.time()
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **options
                )
                answer = (r.choices[0].message.content or "").strip()
                usage = getattr(r, "usage", None)
                n_in = usage.prompt_tokens if usage else 0
                n_out = usage.completion_tokens if usage else 0
                _record_model_call(model, time.time() - start, ok=True)
                return answer, (n_in, n_out)
            except RateLimitError as e:
                pool.mark_rate_limited(key_idx)
                _record_model_call(model, time.time() - start, ok=False, rate_limited=True, error=e)
                last_error = e
                logger.warning(f"Rate limit on {get_provider_for_model(model)} key #{key_idx} (attempt {attempt + 1}), rotating...")
                await asyncio.sleep(1.0)
                continue
            except (AuthenticationError, NotFoundError, PermissionDeniedError) as e:
                _record_model_call(model, time.time() - start, ok=False, error=e)
                logger.error(f"Non-retryable error for model '{model}' on key #{key_idx}: {e}")
                raise
            except BadRequestError as e:
                _record_model_call(model, time.time() - start, ok=False, error=e)
                if "context_length" in str(e).lower():
                    raise
                last_error = e
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                _record_model_call(model, time.time() - start, ok=False, error=e)
                last_error = e
                logger.error(f"API error for model '{model}' on key #{key_idx}: {e}")
                await asyncio.sleep(1.0)
                continue

        raise last_error or Exception(f"All API keys exhausted for model '{model}'")

    async def send_message_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1200
    ) -> AsyncGenerator[Tuple[str, str, Tuple[int, int]], None]:
        """Streams completion tokens with key rotation + optional fallback chain (§27)."""
        options = dict(DEFAULT_COMPLETION_OPTIONS)
        options["temperature"] = temperature
        options["max_tokens"] = max_tokens
        options["stream"] = True

        last_error = None
        for model in self._candidate_models():
            try:
                async for item in self._stream_with_model(model, messages, options):
                    yield item
                if model != self.model:
                    self.used_fallback = True
                    self.used_model = model
                    logger.warning(f"Streaming fallback engaged: '{self.model}' -> '{model}'")
                return
            except BadRequestError:
                raise  # request-shaped problem — other models won't help
            except Exception as e:
                last_error = e
                logger.warning(f"Streaming with model '{model}' failed: {e}. Trying next in chain…")
                continue
        raise last_error or Exception("All models in the fallback chain are unavailable during streaming")

    async def _stream_with_model(
        self,
        model: str,
        messages: List[Dict],
        options: Dict
    ) -> AsyncGenerator[Tuple[str, str, Tuple[int, int]], None]:
        """Single-model streaming with key rotation across that model's provider pool."""
        pool = get_pool_for_model(model)
        last_error = None
        for attempt in range(pool.num_keys + 1):
            client, key_idx = pool._get_next_client()
            start = time.time()
            try:
                r_gen = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **options
                )
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
                        n_input_tokens, n_output_tokens = self._count_tokens_from_messages(
                            messages, answer, model=model
                        )
                        yield "not_finished", answer, (n_input_tokens, n_output_tokens)

                _record_model_call(model, time.time() - start, ok=True)
                yield "finished", answer.strip(), (n_input_tokens, n_output_tokens)
                return  # success, exit retry loop

            except RateLimitError as e:
                pool.mark_rate_limited(key_idx)
                _record_model_call(model, time.time() - start, ok=False, rate_limited=True, error=e)
                last_error = e
                logger.warning(f"Rate limit on {get_provider_for_model(model)} key #{key_idx} during streaming, rotating...")
                await asyncio.sleep(1.0)
                continue
            except (AuthenticationError, NotFoundError, PermissionDeniedError) as e:
                # Rotating keys won't help — the key is bad or the model doesn't exist.
                _record_model_call(model, time.time() - start, ok=False, error=e)
                logger.error(f"Non-retryable error for model '{model}' on key #{key_idx}: {e}")
                raise
            except Exception as e:
                _record_model_call(model, time.time() - start, ok=False, error=e)
                last_error = e
                logger.error(f"Streaming error for model '{model}' on key #{key_idx}: {e}")
                await asyncio.sleep(1.0)
                continue

        raise last_error or Exception(f"All API keys exhausted during streaming for model '{model}'")

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
    """Lightweight helper for background tasks (summarization, memory extraction) with key rotation."""
    target_model = model or config.memory_model
    try:
        pool = get_pool_for_model(target_model)
    except ProviderNotConfiguredError as e:
        logger.warning(f"fast_chat_completion skipped: {e}")
        return ""
    for attempt in range(pool.num_keys + 1):
        client, key_idx = pool._get_next_client()
        try:
            r = await client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=40.0
            )
            return (r.choices[0].message.content or "").strip()
        except RateLimitError:
            pool.mark_rate_limited(key_idx)
            await asyncio.sleep(1.0)
            continue
        except Exception as e:
            logger.warning(f"fast_chat_completion error (key #{key_idx}): {e}")
            await asyncio.sleep(0.5)
            continue
    return ""


async def get_embedding(text: str) -> Optional[List[float]]:
    """Fetches text embedding if an embedding provider is reachable.
    Skips silently for providers that don't support embeddings (e.g. Groq)."""
    if not text or not config.embedding_api_key:
        return None
    if not config.embedding_enabled:
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
    """Transcribes audio using Whisper or speech-to-text API (default provider pool)."""
    if _api_pool is None:
        raise ProviderNotConfiguredError(DEFAULT_PROVIDER)
    client, _ = _api_pool._get_next_client()
    try:
        r = await client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return r.text or ""
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        raise


async def generate_images(prompt: str, n_images: int = 1, size: str = "1024x1024") -> List[bytes]:
    """Generates images using OpenAI/compatible image API (default provider pool)."""
    if _api_pool is None:
        raise ProviderNotConfiguredError(DEFAULT_PROVIDER)
    client, _ = _api_pool._get_next_client()
    r = await client.images.generate(model="gpt-image-1", prompt=prompt, n=n_images, size=size)
    images = [base64.b64decode(item.b64_json) for item in r.data if getattr(item, "b64_json", None)]
    return images
