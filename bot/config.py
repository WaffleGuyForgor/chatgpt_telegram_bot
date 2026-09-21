import os
import yaml
import dotenv
from pathlib import Path

config_dir = Path(__file__).parent.parent.resolve() / "config"

# load yaml config (optional – everything can come from env vars, e.g. on Railway)
config_yaml = {}
_yaml_path = config_dir / "config.yml"
if _yaml_path.exists():
    with open(_yaml_path, 'r', encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f) or {}

# load .env config (optional)
config_env = dotenv.dotenv_values(config_dir / "config.env")
if not config_env and (config_dir.parent / ".env").exists():
    config_env = dotenv.dotenv_values(config_dir.parent / ".env")


def _get(key: str, default=None, cast=None):
    """Resolve a config value: environment variable wins, then yaml, then default."""
    env_key = key.upper()
    if os.environ.get(env_key) is not None:
        value = os.environ[env_key]
    elif config_env.get(env_key) is not None:
        value = config_env[env_key]
    elif key in config_yaml:
        value = config_yaml[key]
    else:
        return default

    if cast is None:
        return value
    return cast(value)


def _str_list(value):
    """Parse a comma-separated string or a yaml list into a list of str/int."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    items = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(int(item))
        except ValueError:
            items.append(item)
    return items


# config parameters
telegram_token = _get("telegram_token", "")

# LLM Gateway credentials (Groq default, OpenRouter fallback)
groq_api_key = _get("groq_api_key", None)
openrouter_api_key = _get("openrouter_api_key", None)
openai_api_key = _get("openai_api_key", None)

# Unified LLM provider settings
# Priority: LLM_API_KEY > GROQ_API_KEY > OPENROUTER_API_KEY > OPENAI_API_KEY
# Supports multiple keys for load spreading: LLM_API_KEYS=sk-abc,sk-def,sk-ghi
_single_key = _get("llm_api_key") or groq_api_key or openrouter_api_key or openai_api_key or ""
_multi_keys_raw = _get("llm_api_keys") or ""  # comma-separated

if _multi_keys_raw:
    llm_api_keys = [k.strip() for k in str(_multi_keys_raw).split(",") if k.strip()]
else:
    llm_api_keys = [_single_key] if _single_key else []

# Keep single key for backward compat
llm_api_key = _single_key
llm_base_url = _get("llm_base_url") or "https://api.groq.com/openai/v1"

# --- Additional LLM providers (per-model routing via models.yml `provider:`) ---

# OpenRouter (supports multiple keys for load spreading: OPENROUTER_API_KEYS=k1,k2,k3)
_openrouter_single_key = _get("openrouter_api_key") or ""
_openrouter_multi_keys = _get("openrouter_api_keys") or ""  # comma-separated
if _openrouter_multi_keys:
    openrouter_api_keys = [k.strip() for k in str(_openrouter_multi_keys).split(",") if k.strip()]
else:
    openrouter_api_keys = [_openrouter_single_key] if _openrouter_single_key else []
openrouter_base_url = _get("openrouter_base_url") or "https://openrouter.ai/api/v1"

# Dahl inference network (supports multiple keys: DAHL_API_KEYS=k1,k2)
_dahl_single_key = _get("dahl_api_key") or ""
_dahl_multi_keys = _get("dahl_api_keys") or ""  # comma-separated
if _dahl_multi_keys:
    dahl_api_keys = [k.strip() for k in str(_dahl_multi_keys).split(",") if k.strip()]
else:
    dahl_api_keys = [_dahl_single_key] if _dahl_single_key else []
dahl_base_url = _get("dahl_base_url") or "https://inference.dahl.global/v1"

# Provider registry: name -> (api keys, base url). "groq" is the default provider.
provider_registry = {
    "groq": {"keys": llm_api_keys, "base_url": llm_base_url},
    "openrouter": {"keys": openrouter_api_keys, "base_url": openrouter_base_url},
    "dahl": {"keys": dahl_api_keys, "base_url": dahl_base_url},
}
import logging as _cfg_log
_cfg_log.getLogger(__name__).info(f"LLM API keys loaded: {len(llm_api_keys)} key(s) for {llm_base_url}")
if llm_api_keys:
    for _i, _k in enumerate(llm_api_keys):
        _masked = _k[:6] + "..." + _k[-4:] if len(_k) > 10 else "???"
        _cfg_log.getLogger(__name__).info(f"  Key #{_i}: {_masked} (len={len(_k)})")
else:
    _cfg_log.getLogger(__name__).warning("NO API KEYS LOADED! Check GROQ_API_KEY or LLM_API_KEY env var.")

_cfg_log.getLogger(__name__).info(
    f"OpenRouter provider: {len(openrouter_api_keys)} key(s) for {openrouter_base_url}"
)
_cfg_log.getLogger(__name__).info(
    f"Dahl provider: {len(dahl_api_keys)} key(s) for {dahl_base_url}"
)

# Bot owner ID (full control panel in DMs)
owner_id = _get("owner_id", 6274319204)
if owner_id is not None:
    owner_id = int(owner_id)

allowed_telegram_usernames = _str_list(_get("allowed_telegram_usernames", []))
new_dialog_timeout = _get("new_dialog_timeout", 600, int)
enable_message_streaming = _get("enable_message_streaming", True, lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes"))
return_n_generated_images = _get("return_n_generated_images", 1, int)
image_size = _get("image_size", "1024x1024")
n_chat_modes_per_page = _get("n_chat_modes_per_page", 5, int)

# Context and memory configurations
max_dialog_messages = _get("max_dialog_messages", 30, int)
rolling_summary_threshold = _get("rolling_summary_threshold", 16, int)
memory_enabled = _get("memory_enabled", True, lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes"))
memory_retrieval_limit = _get("memory_retrieval_limit", 5, int)

# How long an "active object"/topic reference stays usable for follow-ups (§6 expiry)
conversation_state_ttl_minutes = _get("conversation_state_ttl_minutes", 45, int)

# Contextual inline shortcut buttons after substantial answers (§21/§22)
contextual_buttons_enabled = _get("contextual_buttons_enabled", True, lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes"))

# Memory scoring weights (semantic similarity, importance, recency, entity match)
memory_weight_semantic = _get("memory_weight_semantic", 0.45, float)
memory_weight_importance = _get("memory_weight_importance", 0.20, float)
memory_weight_recency = _get("memory_weight_recency", 0.15, float)
memory_weight_entity = _get("memory_weight_entity", 0.20, float)

# Optional embedding API (disabled by default — Groq doesn't support embeddings)
embedding_enabled = _get("embedding_enabled", False, lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes"))
embedding_api_key = _get("embedding_api_key") or llm_api_key
embedding_base_url = _get("embedding_base_url") or llm_base_url
embedding_model = _get("embedding_model", "text-embedding-3-small")

# mongodb: check every Railway / Atlas / docker-compose variable name
_mongo_candidates = [
    "mongodb_uri", "mongo_url", "mongo_private_url", "database_url",
    "MONGO_URL", "MONGODB_URI", "MONGO_PRIVATE_URL", "DATABASE_URL",
    "MONGOHQ_URL", "MONGOLAB_URI", "MONGODB_URL", "DB_URL",
    "MONGO_URI", "MONGOHOST", "MONGO_HOST",
]
mongodb_uri = None

# Log all env vars that contain "mongo" or "database" for debugging
import logging as _cfg_log
_cfg_log.getLogger(__name__).info("Scanning for MongoDB connection variables...")
for _k, _v in os.environ.items():
    _kl = _k.lower()
    if "mongo" in _kl or "database" in _kl or "db_" in _kl:
        # Don't log passwords in full
        _display = _v[:60] + "..." if len(_v) > 60 else _v
        _cfg_log.getLogger(__name__).info(f"  Found env: {_k}={_display}")

for _candidate in _mongo_candidates:
    val = os.environ.get(_candidate) or config_env.get(_candidate)
    if val and val.startswith(("mongodb://", "mongodb+srv://")):
        mongodb_uri = val
        _cfg_log.getLogger(__name__).info(f"MongoDB URI resolved from env var: {_candidate}")
        break

if not mongodb_uri:
    # Docker-compose fallback
    _mongo_port = config_env.get("MONGODB_PORT", os.environ.get("MONGODB_PORT", "27017"))
    mongodb_uri = f"mongodb://mongo:{_mongo_port}"
    _cfg_log.getLogger(__name__).warning(f"No MongoDB env var found! Falling back to docker-compose: {mongodb_uri}")

# chat_modes
with open(config_dir / 'chat_modes.yml', 'r', encoding="utf-8") as f:
    chat_modes = yaml.safe_load(f)

# models
with open(config_dir / 'models.yml', 'r', encoding="utf-8") as f:
    models = yaml.safe_load(f)

# default model – env override, else first available from models.yml
default_model = _get("default_model") or models["available_text_models"][0]
# Validate model exists in models.yml — if not, use first available
if default_model not in models["info"]:
    _cfg_log.getLogger(__name__).warning(f"Model '{default_model}' not found in models.yml, using '{models['available_text_models'][0]}'")
    default_model = models["available_text_models"][0]

_cfg_log.getLogger(__name__).info(f"Default model: {default_model}")

# Background memory extraction & summarization model (falls back to default_model)
memory_model = _get("memory_model") or default_model

# Optional task-based model routing (§26) — disabled by default.
# An explicit /model choice always wins over routing; routing only applies
# to users still on the default model.
routing_enabled = _get("routing_enabled", False, lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes"))
routing_fast_model = _get("routing_fast_model", "")  # serves tiny/short requests when routing is on
routing_deep_model = _get("routing_deep_model", "")  # serves detailed/deep requests when routing is on
for _routed_name, _routed_val in (("routing_fast_model", routing_fast_model), ("routing_deep_model", routing_deep_model)):
    if _routed_val and _routed_val not in models.get("info", {}):
        _cfg_log.getLogger(__name__).warning(f"{_routed_name} '{_routed_val}' not in models.yml — ignoring")
        if _routed_name == "routing_fast_model":
            routing_fast_model = ""
        else:
            routing_deep_model = ""

# Optional fallback chain (§27): ordered list of models tried when the active
# model's provider fails (rate limits, outages). Empty = no automatic fallback.
fallback_models = []
for _fb in _str_list(_get("fallback_models", "")):
    if not isinstance(_fb, str):
        continue
    if _fb in models.get("info", {}):
        fallback_models.append(_fb)
    else:
        _cfg_log.getLogger(__name__).warning(f"Fallback model '{_fb}' not in models.yml — ignoring")
if fallback_models:
    _cfg_log.getLogger(__name__).info(f"Fallback chain: {fallback_models}")

# files
help_group_chat_video_path = Path(__file__).parent.parent.resolve() / "static" / "help_group_chat.mp4"
