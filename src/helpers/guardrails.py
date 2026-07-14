import re
from typing import Any, List, Tuple

import src.config.settings as config


INVALID_QUERY_MESSAGE = "Please enter a valid question about the indexed research papers."
LONG_QUERY_MESSAGE = "The question is too long. Please ask a shorter, focused question."
PROMPT_INJECTION_MESSAGE = (
    "I cannot follow instructions that try to override the system or expose hidden prompts/context. "
    "Please ask a question about the indexed research papers."
)
NO_EVIDENCE_MESSAGE = "I could not find enough relevant evidence in the indexed papers to answer this reliably."
EMPTY_ANSWER_MESSAGE = "The model did not return a usable answer. Please try again."

PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\bforget\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\boverride\s+(the\s+)?(system|developer)\s+(prompt|instructions)\b",
    r"\breveal\s+(the\s+)?(system|developer|hidden)\s+(prompt|instructions|context)\b",
    r"\bshow\s+(the\s+)?(system|developer|hidden)\s+(prompt|instructions|context)\b",
    r"\bprint\s+(the\s+)?(system|developer|hidden)\s+(prompt|instructions|context)\b",
    r"\bact\s+as\s+(a\s+)?(developer|system|admin)\b",
    r"\byou\s+are\s+now\s+(developer|system|admin)\b",
]


def detect_prompt_injection(text: str) -> bool:
    """Detect common attempts to override system instructions or expose hidden context."""
    normalized_text = " ".join((text or "").lower().split())
    return any(re.search(pattern, normalized_text) for pattern in PROMPT_INJECTION_PATTERNS)


def validate_user_query(query: str) -> Tuple[bool, str | None]:
    """Validate the user query before routing or retrieval."""
    if not getattr(config, "ENABLE_GUARDRAILS", True):
        return True, None

    query_text = (query or "").strip()

    if len(query_text) < config.MIN_QUERY_CHARS:
        return False, INVALID_QUERY_MESSAGE

    if len(query_text) > config.MAX_QUERY_CHARS:
        return False, LONG_QUERY_MESSAGE

    if config.ENABLE_PROMPT_INJECTION_GUARDRAIL and detect_prompt_injection(query_text):
        return False, PROMPT_INJECTION_MESSAGE

    return True, None


def has_enough_retrieval_evidence(results: List[Any]) -> bool:
    """Check whether retrieval found enough evidence to justify answer generation."""
    if not getattr(config, "ENABLE_GUARDRAILS", True):
        return True

    return len(results or []) >= config.MIN_RETRIEVAL_RESULTS


def sanitize_retrieved_context(context: str) -> str:
    """Neutralize instruction-like text inside retrieved evidence before it reaches the LLM."""
    if not getattr(config, "ENABLE_GUARDRAILS", True) or not config.ENABLE_CONTEXT_SANITIZATION:
        return context

    sanitized_context = context
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized_context = re.sub(
            pattern,
            "[removed instruction-like text from retrieved context]",
            sanitized_context,
            flags=re.IGNORECASE,
        )

    return sanitized_context


def validate_llm_answer(answer: str) -> Tuple[bool, str | None]:
    """Validate the answer returned by the configured LLM."""
    if not getattr(config, "ENABLE_GUARDRAILS", True):
        return True, None

    if not (answer or "").strip():
        return False, EMPTY_ANSWER_MESSAGE

    return True, None
