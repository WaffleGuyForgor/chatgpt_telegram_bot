# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — 9Router provider (dynamic models from a variable)
- **9Router support**: a generic OpenAI-compatible catch-all provider. Set `NINE_ROUTER_MODELS` (comma-separated model IDs, aliases: `9ROUTER_MODELS`) and they appear in `/model` automatically — no `models.yml` edits needed.
- **Auto-generated display names**: `openrouter/deepseek-v4.1-flash` → *Deepseek v4.1 Flash*, `oc/glm-5.3` → *GLM 5.3*, `claude-opus-4.8` → *claude opus 4.8* (vendor prefix dropped, acronyms uppercased, version tokens preserved; no provider label shown on 9Router buttons).
- `NINE_ROUTER_API_KEY` / `NINE_ROUTER_API_KEYS` (multi-key rotation) and `NINE_ROUTER_BASE_URL` (default `https://api.9router.com/v1`) configure the provider; all existing features (key rotation, fallback chain, routing, /status) work with it.
- Picker buttons now skip models whose callback data would exceed Telegram's 64-byte limit.

### Added — Phase 2 Polish, Stage 3 (conversation intelligence + Telegram-native UX)
- **Active-object tracking with expiry** (`bot/turn_analysis.py` + `context_engine`): "make it shorter" / "the second one" / "do that again" resolve against the previous answer — but only while the state is fresh (default TTL 45 min, `CONVERSATION_STATE_TTL_MINUTES`) and only when the message actually refers to it.
- **Topic transitions**: lightweight topic signatures detect when the user switches subjects, so old context is not injected into unrelated conversations; topic history is kept.
- **Multi-intent messages** are detected and the model is told to answer every part, not just the first sentence.
- **Ambiguity handling**: bare fragments ("make it better") ask *one* short clarifying question only when there is nothing to refer to.
- **Conversation repair**: "no, I meant the other file" triggers a correction hint (acknowledge, apply, don't repeat).
- **Contextual buttons** (§21/§22, sparingly, DMs only): [Shorter] [More detail] [Another version] after detailed answers, [Explain] [Fix] [Optimize] after code — they run as normal conversational follow-ups (`CONTEXTUAL_BUTTONS_ENABLED` to disable).
- **Session chat modes** `/mode`: normal · concise · deep · creative · technical · coding — per-conversation, saved preferences untouched.
- **Conversation summary on demand** (§43): "summarize this conversation" returns a skimmable digest (topic / decisions / open points / action items) with a deterministic fallback if the summarizer model is unavailable.
- New identity rules for fragment resolution, repair behaviour and genuine regeneration variation; tests + harness coverage for all of the above.

### Added — Phase 2 Polish, Stage 2 (memory lifecycle, scoped preferences, routing)
- **Memory lifecycle metadata** (`bot/database.py`): memories now carry `confidence`, `stability`, `usage_count`, `last_confirmed_at` alongside importance/recency, plus `reinforce_memory()` (raise confidence, capped at 1.0) and `weaken_memory()` for contradictions.
- **Memory reinforcement**: when the user restates something already known, the existing memory is reinforced (confidence/usage grow) instead of being duplicated.
- **Confidence-aware retrieval & answers**: retrieval scores are dampened by confidence, and low-confidence memories are marked `[low confidence]` in the prompt with an instruction to hedge rather than state them as fact.
- **Correction-aware extraction**: the memory extractor now prefers updating an existing memory over adding a new one when the user corrects a fact, keeping corrections scoped (e.g. a project correction updates the project memory, not the user's global preferences) and ignoring momentary emotional reactions.
- **Global vs per-conversation preferences** (`bot/personality.py`): stated preferences ("from now on keep it short") update the durable user profile, while one-off reactions ("too long", "don't talk like that", "for this chat be detailed") are scoped to the current conversation and merged with lowest priority below explicit per-message instructions. New "no emojis" preference is honored in the prompt.
- **Preference explanation**: "why did you answer that way?" gets a natural, non-technical explanation of the active preferences (§14).
- **Optional task-based model routing** (`ROUTING_ENABLED`, `ROUTING_FAST_MODEL`, `ROUTING_DEEP_MODEL`): routes tiny/short requests to a fast model and detailed/deep requests to a stronger one — disabled by default, and an explicit `/model` choice always wins. Unconfigured providers fall back to the default model safely.
- **Optional fallback chain** (`FALLBACK_MODELS`, comma-separated): if the active model's provider is exhausted or failing, the next configured model is tried automatically (each with its own provider key pool). The user gets a brief notice that a fallback answered; token accounting is attributed to the model that actually answered.
- **Rolling model performance stats** (calls / errors / rate limits / average latency) recorded per model and shown in the owner panel's detailed stats, along with the active fallback chain.
- New tests for scoped preferences, memory reinforcement/confidence ranking, routing precedence, fallback chain order, and stats accounting.

### Added — Phase 2 Polish, Stage 1 (fast UX, adaptive length, natural tone)
- **Intelligent response length** (`bot/response_tuning.py`): every request is classified as `tiny` / `short` / `normal` / `detailed` / `deep`, which drives the `max_tokens` budget (150 → 4000) and a per-turn length instruction in the system prompt. Explicit user control (*"in one sentence"*, *"detailed explanation"*, *"comprehensive guide"*) always overrides the heuristics.
- **Soft mood adaptation**: each message gets an ephemeral tone signal (`joking` / `frustrated` / `technical` / `casual` / `serious` / `neutral`) injected as an explicitly-marked soft guess — never stored, never treated as fact.
- **Contextual status messages**: the placeholder is picked per request size ("Hmm…", "Looking into it…", "Working on that…", "Putting that together…") instead of a static "Thinking…", once per request (no fake progress).
- **Anti-generic-AI rules**: the base identity now bans filler openers (*Certainly! / Great question!*), per-message greetings, question restating, obligatory conclusions, and *"Let me know if…"* closers; requires honest "I don't know" behavior and natural structural variation.
- **Lightweight response self-check** before delivery (no second model pass): strips banned openers if the model emits them anyway, trims essay-length answers to micro-questions, and proactively detects unbalanced/unsupported HTML so the final message falls back to plain text instead of hitting a Telegram `BadRequest`.

## [2.1.0]

### Added
- **Multi-Provider Model Switching**:
  - Models in `config/models.yml` now declare a `provider:` field (`groq`, `openrouter`, `dahl`) and each request is routed to that provider's own API key pool and base URL.
  - **OpenRouter provider** with free-tier models: `nex-agi/nex-n2.5-pro:free` (⭐ marked as the best/recommended pick), `nvidia/nemotron-3-ultra-550b-a55b:free`, `nex-agi/nex-n2.5-mini:free`.
  - **Dahl provider** (`https://inference.dahl.global/v1`) with `deepseek-ai/DeepSeek-V4-Flash-0731`.
  - **Groq remains the default provider/model** (`openai/gpt-oss-20b`).
  - Multiple API keys per provider for round-robin rotation with rate-limit cooldown: `OPENROUTER_API_KEYS` / `OPENROUTER_API_KEY` and `DAHL_API_KEYS` / `DAHL_API_KEY` (comma-separated), mirroring the existing `LLM_API_KEYS` behavior for Groq.
- **Public Model Picker (`/model`, `/models`)**: anyone (DMs and group members) can switch the active model via an inline keyboard. The picker shows the current model (✅), the recommended best model (⭐ Nex N2.5 Pro), the serving provider per model, and a Persian note: *"اگر مدل به محدودیت (Rate Limit) رسید، لطفاً مدل دیگری را انتخاب کنید."* (if a model is rate-limited, pick another one).
- **Rate-limit friendly errors**: when all keys of a provider are exhausted, the bot replies with a bilingual (English/Persian) message suggesting to switch models via `/model` instead of a generic failure.
- **Provider guard**: selecting or using a model whose provider has no API key configured produces a clear bilingual warning instead of an API crash.
- `/status` now shows the active model's provider and that provider's key-pool health.
- Owner panel's *Switch Model* list now labels each model with its provider.
- New tests covering model→provider routing, models.yml provider integrity, the Groq default, and the ⭐ recommended flag.

### Changed
- `openai_utils` now maintains one `APIKeyPool` per provider instead of a single global pool; `ChatGPT.send_message`, `send_message_stream`, and `fast_chat_completion` resolve the correct pool from the target model.
- Audio transcription and image generation keep using the default (Groq) pool, falling back to any configured provider pool.
- Updated `config/config.example.env` and `config/config.example.yml` with the new provider variables.

## [2.0.0]

### Added
- **Mira-Style Personal AI Assistant Architecture**:
  - **Natural Language First Controls**: Persistent memory operations without requiring rigid slash commands. Supports natural statements:
    - *"Remember that [fact/preference]"* — extracts and stores durable context.
    - *"What do you remember about me?"* — surfaces human-readable memory summary.
    - *"Forget [topic/fact]"* — selectively removes memories.
    - *"Don't remember this"* / *"Off the record"* — conversation turn bypasses long-term memory extraction.
    - *"Stop remembering things"* / *"Resume memory"* — enables or pauses memory learning.
  - **Structured Long-Term Memory Engine (`bot/memory.py`)**:
    - Multi-category schema: `identity`, `preference`, `project`, `entity`, `decision`, `personal`, `topic`.
    - Automated conflict resolution: updates contradicting memories (e.g. upgraded OS, switched preferences) instead of maintaining conflicting truths.
    - Configurable weighted retrieval scoring:
      `score = (0.45 * semantic) + (0.20 * importance) + (0.15 * recency) + (0.20 * entity_match)`.
    - Strict anti-prompt-injection isolation delimiters (`<relevant_memory>`) with instructions treating memories strictly as contextual reference data, not executable instructions.
  - **Rolling Conversation Summaries (`bot/context_engine.py`)**:
    - Dynamic summarization of older conversation turns when dialog history exceeds thresholds, retaining recent context and preventing context-window overflow.
  - **Group Conversation Engine (`bot/group_engine.py`)**:
    - Speaker tracking with `[Sender Name]: message` formatting and reply relationship preservation.
    - Natural *"What did I miss?"* / *"What did we decide?"* group catch-up digests summarizing recent decisions, topics, and action items.
    - Strict context isolation: group memories are scoped to `group:{chat_id}` and never leak into private user memories or vice versa.
    - `/sethome` command to designate a Home Chat with optional active auto-reply mode.
  - **Multimodal Ingestion Pipeline (`bot/multimodal.py`)**:
    - Voice message handling with Whisper transcription piped directly into conversational context.
    - Vision and image handling with base64 conversion and conversational state preservation for natural follow-up modifications (*"make it darker"*, *"what is in the top right?"*).
    - Document parsing for PDF files (via `pypdf`), text files, Markdown, JSON, CSV, and source code files.
  - **Personality & Silent Style Adaptation (`bot/personality.py`)**:
    - Direct, natural, human-like responses without corporate boilerplate, self-introductions, or *"As an AI assistant..."* disclaimers.
    - Dynamic silent adaptation to user preferences (verbosity, formality, technical depth).
  - **Owner Control Panel (`bot/bot.py`)**:
    - Interactive inline dashboard in private DMs (`/panel`, `/admin`, `/owner`) for owner ID `6274319204`.
    - Real-time bot analytics (users, groups, dialogs, durable memories).
    - Home chat manager with auto-reply toggling.
    - Global model switching on the fly.
    - Broadcast announcement tool (`/broadcast <message>`) to all registered users and groups with delivery counters.
  - **Railway Deployment Infrastructure**:
    - `railway.json` and `Procfile` configuration.
    - Multi-variable MongoDB connection fallback (`MONGODB_URI`, `MONGO_URL`, `MONGO_PRIVATE_URL`, `DATABASE_URL`).
    - Standardized production container entrypoint in `Dockerfile`.
  - Comprehensive unit test suite (`tests/test_mira_assistant.py`) for intent detection, memory scoring, style adaptation, group digests, and message chunking.

### Changed
- Default model updated to `nex-agi/nex-n2.5-pro:free` routed via OpenRouter.
- Unified OpenRouter / 9Router / OpenAI-compatible API gateway client with exponential backoff retries.
- Re-architected message dispatch to support asynchronous, non-blocking background workers for memory extraction and summarization.
- Response streaming now uses adaptive throttling to respect Telegram rate limits while editing messages in real time.
- Long response splitting gracefully partitions responses exceeding 4,000 characters along paragraph and newline boundaries without truncating code blocks.
- Improved error handling to provide friendly, conversational error notices instead of dumping technical stack traces to users.

### Security
- Verified zero API keys, tokens, or credentials in tracked git files.
- Strengthened `.gitignore` to strictly exclude `config/config.yml`, `config/config.env`, `.env`, `node_modules/`, and Railway configuration caches.

## [1.3.1]

### Fixed
- In group chats the bot no longer replies "I don't know how to read files or
  videos" to videos and documents it wasn't mentioned in — `unsupport_message_handle`
  now respects the same `is_bot_mentioned` check as the other handlers (#454).

### Security
- Bumped `python-dotenv` 1.0.1 → 1.2.2 to resolve a symlink-following advisory in
  `set_key` (Dependabot).

## [1.3.0]

### Removed
- Legacy `gpt-3.5-turbo` and `gpt-4` models from the menu and config — the
  lineup is now gpt-4o-mini (default), gpt-5.5, gpt-4o and the Claude models.

### Changed
- Simplified the token-count overhead table (all current chat models share
  the same overhead).
- Refreshed the README (title and feature list) to drop legacy GPT-4 mentions.

### Fixed
- `/balance` no longer raises a `KeyError` for users with historical usage of
  models that have since been removed from the config.

## [1.2.0]

### Added
- Minimal GitHub Actions CI: installs requirements on Python 3.12 and
  byte-compiles the bot on every push and pull request.

### Changed
- Image generation now uses OpenAI **gpt-image-1** (default size 1024×1024);
  images are returned as bytes and `/balance` / pricing updated accordingly.
- Updated `gpt-4o` pricing to current rates (0.0025 / 0.01 per 1k tokens).

### Removed
- Deprecated models `gpt-3.5-turbo-16k`, `gpt-4-1106-preview` and
  `gpt-4-vision-preview` from the model menu (gpt-4o / gpt-4o-mini cover them).

## [1.1.0]

### Added
- **OpenRouter provider support**: models can declare `provider: openrouter`
  in `models.yml` and are routed through an OpenAI-compatible OpenRouter
  client. New `openrouter_api_key` / `openrouter_api_base` config options.
- **Anthropic Claude models** via OpenRouter: Claude Opus 4.8, Claude Sonnet
  and Claude Haiku.
- OpenAI `gpt-5.5` (via OpenRouter).

### Changed
- Chat models are now dispatched by their `type` in `models.yml` instead of
  hardcoded model-name lists, so adding a model is a config-only change.
- Token counting falls back to the `o200k_base` encoding for models unknown
  to `tiktoken` (e.g. Claude).
- The `/settings` model picker lays buttons out in rows of two to stay within
  Telegram's per-row inline-button limit.
- Image understanding is now driven by a `vision: true` flag in `models.yml`,
  so any vision-capable model (GPT-4o, GPT-4o mini, GPT-5.5, Claude) can read
  images — no longer limited to GPT-4o / GPT-4 Vision.

## [1.0.0]

### Added
- `gpt-4o` and `gpt-4o-mini` models, with `gpt-4o-mini` as the new default.
- `VERSION` file and this changelog.

### Changed
- Migrated from the deprecated `openai==0.28` API to the `openai>=1.x` SDK
  (`AsyncOpenAI` client, new chat/image/audio/moderation methods, updated
  error classes).
- Default model is now `gpt-4o-mini` instead of the outdated `gpt-3.5-turbo`
  (new users, `/new`, and the `ChatGPT` fallback).
- Upgraded the Docker base image to Python 3.12 (3.8 is end-of-life).
- Bumped dependencies: `python-telegram-bot` 20.8, `pymongo` 4.6.3,
  `PyYAML` 6.0.2, `python-dotenv` 1.0.1, `tiktoken` >= 0.7.0.
- Docker: smaller image (`pip --no-cache-dir`), unbuffered logs
  (`PYTHONUNBUFFERED`), live-mounted config, expanded `.dockerignore`,
  removed the obsolete Compose `version` key.

### Removed
- `text-davinci-003` and its legacy completion code paths (the model was
  shut down by OpenAI).

### Fixed
- Corrected `gpt-4o` pricing and scores in `models.yml`.
- `tiktoken` now recognizes the `o200k_base` encoding used by `gpt-4o`.
- Removed debug logging that fired on every incoming message.
- Narrowed bare `except:` clauses so shutdown/cancellation propagates.
- Stopped tracking `.DS_Store`; various typo and docs fixes.
