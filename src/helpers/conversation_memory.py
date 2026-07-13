"""Utilities for maintaining the bounded conversational-memory window."""

from typing import Dict, List, Sequence

import src.config.settings as config

def retain_recent_completed_turns(
    chat_history: Sequence[Dict[str, str]] | None,
    max_turns: int = config.MAX_CONVERSATION_TURNS,
) -> List[Dict[str, str]]:
    """Keep the latest completed user/assistant exchanges in chronological order.

    The chat UI passes completed exchanges as alternating user and assistant
    messages, so four turns are simply the last eight history messages.
    """
    if not chat_history or max_turns <= 0:
        return []

    message_limit = max_turns * 2
    return list(chat_history[-message_limit:])
