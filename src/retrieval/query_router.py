import re
import json
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

import src.config.settings as config
from src.helpers.llm_helper import call_configured_llm
from src.helpers.conversation_memory import retain_recent_completed_turns
from src.prompts.prompts import STATIC_GREETING_RESPONSE, ROUTER_SYSTEM_INSTRUCTION

class QueryIntentSchema(BaseModel):
    requires_retrieval: bool = Field(
        description="True if the user query requires facts, definitions, or context from research papers, false if it is a greeting or general chit-chat."
    )
    direct_response: Optional[str] = Field(
        default=None,
        description="The direct conversational answer to provide to the user if requires_retrieval is false, otherwise null."
    )

class QueryRouter:
    def __init__(self):
        # Tier 1 patterns anchored to match ONLY pure, standalone greetings/chit-chat
        
        self.pure_chit_chat_patterns = [
            r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\s*[!.,]?\s*$",
            r"^\s*(who\s+are\s+you|what\s+are\s+you|your\s+name)\s*[?.,]?\s*$",
            r"^\s*(thanks|thank\s+you|thx)\s*[!.,]?\s*$",
            r"^\s*(bye|goodbye|see\s+ya)\s*[!.,]?\s*$"
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.pure_chit_chat_patterns]
        self.response_refinement_patterns = [
            r"\b(shorter|brief|concise|summari[sz]e|compress)\b",
            r"\b(rewrite|rephrase|simplify|format|convert)\b",
            r"\b(in|within)\s+\d+\s*[-–]?\s*\d*\s*(lines?|sentences?|bullets?|points?)\b",
            r"\b\d+\s*[-–]?\s*\d*\s*(lines?|sentences?|bullets?|points?)\b",
            r"\b(as|into)\s+(bullet|bullets|table|paragraph|points?)\b",
            r"\b(previous|above|last)\s+(answer|response|explanation)\b",
        ]
        self.compiled_refinement_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.response_refinement_patterns
        ]

    def _matches_greetings_regex(self, query: str) -> bool:
        """Checks if the query is 100% just a standard greeting or chit-chat phrase."""
        return any(pattern.match(query) for pattern in self.compiled_patterns)

    def is_response_refinement_request(self, query: str, chat_history: List[dict] = None) -> bool:
        """Detect follow-ups that ask to rewrite or reformat the previous assistant answer."""
        if not chat_history:
            return False

        has_previous_answer = any(message.get("role") == "assistant" for message in chat_history)
        if not has_previous_answer:
            return False

        return any(pattern.search(query or "") for pattern in self.compiled_refinement_patterns)

    def resolve_query_intent(self, query: str, chat_history: List[dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Two-tier router using structured JSON parsing:
        - Tier 1: Pure static response for standalone greetings (No LLM).
        - Tier 2: LLM router returning structured JSON via Pydantic validation.
        """
        chat_history = retain_recent_completed_turns(chat_history)

        print(f"\nEvaluating incoming query for routing: '{query}'")
        
        # Tier 1: Pure standalone greeting check -> Return static text instantly
        if self._matches_greetings_regex(query):
            print("[QueryRouter] Tier 1 (Regex) triggered: Standalone greeting detected. Returning static response.")
            return False, STATIC_GREETING_RESPONSE

        # Tier 2: Escalates to LLM with JSON response format enforcement
        print("[QueryRouter] Tier 1 passed (not a standalone greeting). Escalating to Tier 2 (LLM Structured Router)...")
        
        json_instruction = (
            f"{ROUTER_SYSTEM_INSTRUCTION}\n"
            "You MUST respond with a valid JSON object matching this schema:\n"
            '{"requires_retrieval": boolean, "direct_response": string or null}'
        )
        
        messages = [{"role": "system", "content": json_instruction}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        try:
            raw_output = call_configured_llm(
                model_name=config.LLM_CLASSIFIER_MODEL_NAME,
                messages=messages,
                temperature=config.LLM_CLASSIFIER_TEMPERATURE,
                max_tokens=256,
                response_format={"type": "json_object"}
            ).strip()
            
            print(f"[QueryRouter] Tier 2 raw JSON output: '{raw_output}'")
            
            # Validate output cleanly via Pydantic
            parsed_data = QueryIntentSchema.model_validate_json(raw_output)
            
            if not parsed_data.requires_retrieval:
                direct_ans = parsed_data.direct_response or STATIC_GREETING_RESPONSE
                print("[QueryRouter] Tier 2 decided: Non-retrieval (parsed via JSON). Using direct answer.")
                return False, direct_ans
            else:
                print("[QueryRouter] Tier 2 decided: Retrieval required.")
                return True, None
                
        except Exception as e:
            print(f"[QueryRouter Warning] Structured JSON validation failed, defaulting to retrieval: {e}")
            return True, None
