"""
Turn analysis: follow-up detection, active-object freshness, topic transitions,
multi-intent counting, ambiguity checks, repair detection, and contextual buttons.

All heuristics run locally — no extra model calls.
"""
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import config

# =====================================================================
# Tokenisation helpers
# =====================================================================

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "are", "was",
    "can", "could", "would", "should", "how", "what", "why", "when", "where",
    "who", "its", "it's", "but", "not", "have", "has", "had", "does", "did",
    "from", "into", "about", "there", "then", "than", "them", "they", "our",
}


def content_tokens(text: str) -> set:
    """Lowercase content words (3+ chars, stopwords removed)."""
    return {t for t in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", (text or "").lower())
            if t not in _STOPWORDS}


# =====================================================================
# Follow-up / active object handling (§5, §6)
# =====================================================================

FOLLOWUP_MARKERS = [
    "shorter", "longer", "make it", "make that", "change that", "change it",
    "the second", "the first", "the third", "do that again", "another version",
    "try again", "regenerate", "rewrite", "same thing", "again", "instead",
    "that one", "this one", "the other", "what about", "and now", "continue",
    "go on", "summari", "expand", "more of that", "less of that", "darker",
    "lighter", "bigger", "smaller", "the last", "that version",
]

# Bare fragments that only make sense with an active object
BARE_FRAGMENTS = {
    "yeah", "yep", "nah", "no", "yes", "ok", "okay", "fine", "go on",
    "continue", "do it", "do that", "again", "try again", "same", "that one",
    "this one", "the second one", "the first one", "why", "how come", "why though",
    "hmm", "really", "sure", "wait", "actually no", "the second", "the first",
}

# Fragments that are ambiguous when there is no active object to refer to (§9)
AMBIGUOUS_WITHOUT_OBJECT = [
    "make it better", "make this better", "make that better", "improve it",
    "fix it", "change it", "do it", "make it nicer", "make it shorter",
    "make it longer", "redo it", "try again", "another version",
]


def last_output_is_fresh(state: Dict, ttl_minutes: Optional[int] = None) -> bool:
    """True when the stored conversation state is still within its TTL (§6 expiry)."""
    ttl = config.conversation_state_ttl_minutes if ttl_minutes is None else ttl_minutes
    updated = state.get("updated_at") if state else None
    if not isinstance(updated, datetime):
        return True  # unknown timestamp — don't punish the user
    return (datetime.now() - updated) <= timedelta(minutes=ttl)


def looks_like_followup(text: str, last_output: str = "", last_query: str = "") -> bool:
    """
    Detects conversational fragments that refer to the previous turn
    ("make it shorter", "the second one", "do that again", "yeah") (§5, §6).
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.rstrip(".!?") in BARE_FRAGMENTS:
        return True
    if len(t.split()) <= 12 and any(m in t for m in FOLLOWUP_MARKERS):
        return True

    # Lexical overlap with the previous output/query
    prev_tokens = content_tokens(last_output) | content_tokens(last_query)
    cur_tokens = content_tokens(text)
    if not prev_tokens or not cur_tokens:
        return False
    overlap = cur_tokens & prev_tokens
    if len(overlap) >= 2:
        return True
    return len(overlap) / max(1, len(cur_tokens)) >= 0.5


def should_use_active_object(text: str, state: Dict, last_output: str) -> bool:
    """Gate for injecting the previous answer as an explicit reference object."""
    if not last_output:
        return False
    if not last_output_is_fresh(state):
        return False
    return looks_like_followup(text, last_output, state.get("last_user_query", ""))


# =====================================================================
# Topic transitions (§7)
# =====================================================================

def extract_topic_keywords(text: str, limit: int = 6) -> List[str]:
    """Cheap topic signature: most meaningful words of the message."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{4,}\b", (text or "").lower())
    keywords: List[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or tok in keywords:
            continue
        keywords.append(tok)
        if len(keywords) >= limit:
            break
    return keywords


def detect_topic_shift(text: str, state: Dict) -> bool:
    """
    True when the new message clearly starts a different topic than the active one,
    so stale context is not injected into an unrelated conversation (§7).
    """
    if not state:
        return True
    active_topic = state.get("active_topic_keywords") or []
    if not active_topic:
        return True

    cur_tokens = content_tokens(text)
    if not cur_tokens:
        return False
    if set(active_topic) & cur_tokens:
        return False
    # Also consider the previous query/output before declaring a shift
    prev_tokens = content_tokens(state.get("last_user_query", ""))
    if prev_tokens & cur_tokens:
        return False
    return not looks_like_followup(text, state.get("last_output", ""), state.get("last_user_query", ""))


# =====================================================================
# Multi-intent messages (§8)
# =====================================================================

_CLAUSE_SPLIT_RE = re.compile(r"[,;?]|\band\b|\balso\b|\bplus\b", re.IGNORECASE)


def split_intents(text: str, max_intents: int = 5) -> List[str]:
    """
    Splits a message into its distinct asks, e.g.
    "fix this config, explain why it broke, and tell me if the model is still using 9Router"
    -> 3 intents.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    clauses = []
    for part in _CLAUSE_SPLIT_RE.split(raw):
        part = (part or "").strip(" .!?-")
        if len(part.split()) >= 3:
            clauses.append(part)
    if not clauses:
        return [raw]
    return clauses[:max_intents]


def count_intents(text: str) -> int:
    """Approximate number of separate requests in one message (§8)."""
    intents = split_intents(text)
    return max(1, len(intents))


# =====================================================================
# Ambiguity handling (§9)
# =====================================================================

def needs_clarification(text: str, has_active_object: bool) -> bool:
    """
    Only ask when ambiguity materially affects the answer:
    a bare 'make it better'-style fragment with nothing to refer to.
    """
    if has_active_object:
        return False
    t = (text or "").strip().lower().rstrip(".!?")
    if not t:
        return False
    if t in BARE_FRAGMENTS and len(t.split()) <= 4:
        return True
    return any(t == p or t.startswith(p + " ") for p in AMBIGUOUS_WITHOUT_OBJECT)


# =====================================================================
# Regeneration, repair, summaries (§18, §23, §43)
# =====================================================================

_REGENERATE_RE = re.compile(r"\b(regenerate|try again|another version|different version|redo that|do that again|start over|new version)\b", re.IGNORECASE)
_REPAIR_RE = re.compile(r"\b(no,? i meant|i meant the other|that'?s not what i|not that one|no i meant|i meant|wrong one|you misunderstood|that'?s wrong)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"\b(summari[sz]e (this|our|the) (conversation|chat|thread)|recap (this|our|the) (conversation|chat)|what did we (talk|discuss) about|catch me up on this chat)\b", re.IGNORECASE)


def is_regenerate_request(text: str) -> bool:
    return bool(_REGENERATE_RE.search(text or ""))


def is_repair_message(text: str) -> bool:
    return bool(_REPAIR_RE.search(text or ""))


def is_conversation_summary_request(text: str) -> bool:
    return bool(_SUMMARY_RE.search(text or ""))


# =====================================================================
# Chat modes (§25)
# =====================================================================

CHAT_MODES = {
    "normal": "",
    "concise": "Session mode: concise — keep every answer short and to the point.",
    "deep": "Session mode: deep — give thorough answers with explicit reasoning.",
    "creative": "Session mode: creative — explore ideas, offer options and angles.",
    "technical": "Session mode: technical — be precise, include concrete details and trade-offs.",
    "coding": "Session mode: coding — code first, minimal prose, production-minded.",
}
DEFAULT_CHAT_MODE = "normal"


def mode_instruction(mode: Optional[str]) -> str:
    if not mode:
        return ""
    return CHAT_MODES.get(mode, "")


# =====================================================================
# Contextual buttons (§21, §22)
# =====================================================================

FOLLOWUP_INSTRUCTIONS = {
    "shorter": "Make that shorter and tighter while keeping the key points.",
    "more": "Give me a more detailed version of that, with the important specifics.",
    "regenerate": "Give me a different version of that answer — vary the wording, structure and angle, but keep the same task.",
    "explain": "Explain that code clearly, step by step.",
    "fix": "Find and fix any problems in that code.",
    "optimize": "Optimize that code for performance and readability.",
}


def contextual_buttons(size: str, has_code: bool = False) -> List[List[Tuple[str, str]]]:
    """
    Returns inline-button rows as (label, action) pairs.
    Buttons are shortcuts for normal conversational follow-ups — used sparingly.
    """
    if has_code:
        return [[("🧠 Explain", "explain"), ("🛠 Fix", "fix"), ("⚡ Optimize", "optimize")]]
    if size in ("detailed", "deep"):
        return [[("✂️ Shorter", "shorter"), ("📖 More detail", "more"), ("🔁 Another version", "regenerate")]]
    return []


def answer_contains_code(answer: str) -> bool:
    return "```" in (answer or "")
