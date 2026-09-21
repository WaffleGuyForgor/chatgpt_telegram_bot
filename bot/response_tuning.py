"""
Response tuning: request-size classification, soft mood detection,
contextual status messages, and lightweight response self-checks.

Everything here is heuristic and runs locally — no extra model calls.
"""
import random
import re
from typing import Optional

# =====================================================================
# 1. REQUEST SIZE CLASSIFICATION (tiny / short / normal / detailed / deep)
# =====================================================================

SIZE_TINY = "tiny"
SIZE_SHORT = "short"
SIZE_NORMAL = "normal"
SIZE_DETAILED = "detailed"
SIZE_DEEP = "deep"

# max_tokens budget per size class
_SIZE_MAX_TOKENS = {
    SIZE_TINY: 150,
    SIZE_SHORT: 450,
    SIZE_NORMAL: 1200,
    SIZE_DETAILED: 2500,
    SIZE_DEEP: 4000,
}

# Explicit user control over length always wins (Master prompt §2)
_EXPLICIT_TINY = [
    "one word", "yes or no", "in one sentence", "single sentence",
    "answer in one line", "one liner", "one-liner",
]
_EXPLICIT_SHORT = [
    "short answer", "answer briefly", "keep it brief", "be brief",
    "quick answer", "tl;dr", "in short", "sum it up",
]
_EXPLICIT_DETAILED = [
    "in detail", "detailed explanation", "explain deeply", "deep dive",
    "thorough explanation", "elaborate", "walk me through",
]
_EXPLICIT_DEEP = [
    "comprehensive guide", "complete guide", "full guide", "exhaustive",
    "everything about", "from scratch", "end to end", "end-to-end",
]

# Complexity signals
_DEEP_SIGNALS = [
    "design an entire", "design a full", "design a complete", "architect",
    "entire system", "full architecture", "complete system", "whole system",
    "production-ready", "production ready", "full implementation",
]
_DETAILED_SIGNALS = [
    "compare", "versus", " vs ", "pros and cons", "difference between",
    "differences between", "analyze", "analyse", "trade-offs", "tradeoffs",
    "advantages and disadvantages", "evaluate", "review this", "critique",
    "step by step", "explain why", "debug",
    "refactor", "optimize", "write a function", "write a script",
    "implement", "build a", "create a",
]

_MATH_RE = re.compile(r"^[\d\s+\-*/x×÷^().=?,!%]+$")
_FRAGMENT_WORDS = {
    "yeah", "yep", "yup", "nah", "nope", "ok", "okay", "k", "kk",
    "yes", "no", "sure", "fine", "lol", "lmao", "haha", "hmm", "hm",
    "thanks", "thank you", "thx", "ty", "cool", "nice", "wow",
    "hi", "hey", "hello", "yo", "sup", "hola",
}
_QUESTION_STARTERS = ("what", "who", "when", "where", "which", "why", "how",
                      "is", "are", "was", "were", "can", "could", "does", "do")


def classify_request_size(text: str, has_document: bool = False, has_image: bool = False) -> str:
    """
    Classifies the request approximately as tiny / short / normal / detailed / deep.
    Explicit user instructions always take priority over heuristics.
    """
    t = (text or "").strip().lower()
    words = t.split()
    n_words = len(words)

    # --- Explicit overrides ---
    if any(p in t for p in _EXPLICIT_TINY):
        return SIZE_TINY
    if any(p in t for p in _EXPLICIT_DEEP):
        return SIZE_DEEP
    if any(p in t for p in _EXPLICIT_DETAILED):
        return SIZE_DETAILED
    if any(p in t for p in _EXPLICIT_SHORT):
        return SIZE_SHORT

    # --- Deep: big design/architecture asks or very long multi-part prompts ---
    if any(p in t for p in _DEEP_SIGNALS):
        return SIZE_DEEP
    if len(t) > 700 or t.count("\n\n") >= 3:
        return SIZE_DEEP

    # --- Detailed: comparisons, analysis, code tasks, multi-intent ---
    if any(p in t for p in _DETAILED_SIGNALS):
        return SIZE_DETAILED
    if t.count("?") >= 2 or ("```" in t):
        return SIZE_DETAILED

    # --- Tiny: greetings, fragments, pure math, micro questions ---
    if t in _FRAGMENT_WORDS:
        size = SIZE_TINY
    elif _MATH_RE.match(t) and any(ch.isdigit() for ch in t):
        size = SIZE_TINY
    elif (n_words <= 7 and t.endswith("?") and t.startswith(_QUESTION_STARTERS)
          and not any(p in t for p in ("how does", "how do ", "why does", "why do "))):
        # e.g. "what is 9×9?" / "what is the capital of france?"
        size = SIZE_TINY
    # --- Short: simple factual/how questions ---
    elif n_words <= 14 and t.startswith(_QUESTION_STARTERS):
        size = SIZE_SHORT
    else:
        size = SIZE_NORMAL

    # Attachments imply at least a normal analysis task
    if (has_document or has_image) and size in (SIZE_TINY, SIZE_SHORT):
        size = SIZE_NORMAL
    return size


def max_tokens_for_size(size: str) -> int:
    return _SIZE_MAX_TOKENS.get(size, _SIZE_MAX_TOKENS[SIZE_NORMAL])


def length_instruction(size: str) -> str:
    """System-prompt hint so the model picks the right answer length itself."""
    return {
        SIZE_TINY: "Response length: this is a micro-question — answer in one short sentence or less. No preamble, no elaboration.",
        SIZE_SHORT: "Response length: keep it to a few sentences — direct answer first, minimal extra context.",
        SIZE_NORMAL: "Response length: a normal conversational answer — complete but not padded.",
        SIZE_DETAILED: "Response length: this needs a thorough, well-structured answer — cover the important angles, use structure where it helps.",
        SIZE_DEEP: "Response length: this is a deep design/architecture request — take the space needed for a complete, structured, production-minded answer.",
    }.get(size, "")


# =====================================================================
# 2. SOFT MOOD / STYLE DETECTION (never treated as certainty)
# =====================================================================

MOOD_JOKING = "joking"
MOOD_FRUSTRATED = "frustrated"
MOOD_TECHNICAL = "technical"
MOOD_CASUAL = "casual"
MOOD_SERIOUS = "serious"
MOOD_NEUTRAL = "neutral"

_FRUSTRATED_SIGNALS = [
    "wtf", "wth", "why isn't this working", "why is this not working",
    "still broken", "not working again", "broken again", "this is broken",
    "ugh", "damn", "ffs", "hate this", "so stupid", "piece of junk",
    "doesn't work", "doesnt work", "keeps failing", "giving up",
]
_JOKING_SIGNALS = ["lol", "lmao", "haha", "hehe", "😂", "🤣", "jk", "just kidding", "kidding"]
_TECHNICAL_SIGNALS = [
    "error", "exception", "stack trace", "traceback", "segfault",
    "nullpointer", "undefined", "nan", "500", "404", "401", "403",
    "config", "deploy", "docker", "kubernetes", "api key", "endpoint",
    "function", "class ", "import ", "npm ", "pip ", "git ", "sql",
    "regex", "compile", "runtime", "async", "await",
]
_SERIOUS_SIGNALS = [
    "serious question", "be serious", "important", "urgent",
    "i need your honest", "no jokes", "this is serious",
]
_CASUAL_SIGNALS = ["yo", "sup", "hey hey", "wassup", "wya", "how's it going", "hows it going"]


def detect_conversation_mood(text: str) -> str:
    """
    Infers the current conversational style from the message.
    This is a SOFT signal — it must never be treated as a fact about feelings.
    """
    t = (text or "").strip().lower()
    if not t:
        return MOOD_NEUTRAL

    # Serious beats joking (someone can joke then get serious)
    if any(p in t for p in _SERIOUS_SIGNALS):
        return MOOD_SERIOUS
    if any(p in t for p in _FRUSTRATED_SIGNALS):
        return MOOD_FRUSTRATED
    # Heavy punctuation / shouting suggests frustration
    if t.count("!") >= 3 or (len(t) > 12 and t.upper() == text and sum(c.isalpha() for c in t) > 8):
        return MOOD_FRUSTRATED
    if any(p in t for p in _JOKING_SIGNALS):
        return MOOD_JOKING
    if "```" in t or any(p in t for p in _TECHNICAL_SIGNALS):
        return MOOD_TECHNICAL
    words = t.split()
    if words and all(w in _CASUAL_SIGNALS or len(w) <= 4 for w in words) and len(words) <= 5 and not t.endswith("?"):
        return MOOD_CASUAL
    return MOOD_NEUTRAL


def mood_instruction(mood: str) -> str:
    """Soft tone hint for the system prompt (explicitly marked as a guess)."""
    return {
        MOOD_JOKING: "Tone hint (soft guess, may be wrong): the user is joking around — a light, playful reply fits.",
        MOOD_FRUSTRATED: "Tone hint (soft guess, may be wrong): the user seems frustrated — be direct, calm, skip jokes and fluff.",
        MOOD_TECHNICAL: "Tone hint (soft guess, may be wrong): the user is in technical mode — prioritize precise, correct technical detail.",
        MOOD_CASUAL: "Tone hint (soft guess, may be wrong): casual chat — keep it conversational, no essay-style answer.",
        MOOD_SERIOUS: "Tone hint (soft guess, may be wrong): serious conversation — reduce jokes and heavy formatting.",
    }.get(mood, "")

# =====================================================================
# 3. CONTEXTUAL STATUS MESSAGES (coarse progress, no fake detail)
# =====================================================================

_STATUS_POOLS = {
    SIZE_TINY: ["Thinking…", "Hmm…"],
    SIZE_SHORT: ["Thinking…", "Looking into it…"],
    SIZE_NORMAL: ["Thinking…", "On it…", "Working on that…"],
    SIZE_DETAILED: ["Working on that…", "Putting that together…"],
    SIZE_DEEP: ["Working on that…", "This needs a moment…", "Putting that together…"],
}


def pick_status_message(size: str) -> str:
    """One contextual status per request — picked once, not rotated every second."""
    return random.choice(_STATUS_POOLS.get(size, _STATUS_POOLS[SIZE_NORMAL]))


# =====================================================================
# 4. LIGHTWEIGHT RESPONSE SELF-CHECK (no second model pass)
# =====================================================================

# Templated openers that make the bot sound generic (Master prompt §4)
_BANNED_OPENERS = [
    "certainly!", "absolutely!", "of course!", "sure!", "sure thing!",
    "great question!", "good question!", "that's a great question",
    "i'd be happy to help", "i'd be happy to assist", "happy to help",
    "as an ai language model", "as an ai assistant", "as an ai,",
    "certainly.", "absolutely.", "definitely!",
]

_TELEGRAM_TAGS = ("b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
                  "code", "pre", "a", "tg-spoiler", "blockquote", "span")
_TAG_RE = re.compile(r"<(/?)([a-zA-Z-]+)(?:\s[^>]*)?>")


def html_is_balanced(text: str) -> bool:
    """Checks that Telegram-supported HTML tags are properly opened/closed."""
    stack = []
    for m in _TAG_RE.finditer(text or ""):
        closing, tag = m.group(1) == "/", m.group(2).lower()
        if tag not in _TELEGRAM_TAGS:
            return False  # unsupported tag — safer to send as plain text
        if closing:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
        else:
            stack.append(tag)
    return not stack


def strip_banned_opener(answer: str) -> str:
    """Removes templated AI openers if the model produced one anyway."""
    if not answer:
        return answer
    lower = answer.lower().lstrip()
    for opener in _BANNED_OPENERS:
        if lower.startswith(opener):
            cut = len(answer) - len(lower) + len(opener)
            stripped = answer[cut:].lstrip(" \n,.!—:-")
            if stripped:
                return stripped[0].upper() + stripped[1:]
            return stripped
    return answer


def self_check_answer(answer: str, size: str = SIZE_NORMAL) -> str:
    """
    Fast local corrections before delivery:
    - empty/garbage guard
    - strips templated openers the system prompt bans
    - trims absurd overruns for micro-questions
    (HTML safety is handled separately via html_is_balanced.)
    """
    if not answer:
        return answer
    answer = strip_banned_opener(answer.strip())
    if size == SIZE_TINY and len(answer) > 600:
        # A one-word question should not produce an essay — keep first block
        answer = answer.split("\n\n")[0][:600]
    return answer

