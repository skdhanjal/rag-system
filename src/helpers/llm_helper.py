from typing import List, Dict
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential
import src.config.settings as config

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_configured_llm(
    model_name: str, 
    messages: List[Dict[str, str]], 
    temperature: float, 
    response_format: str = None,
    max_tokens: int= config.LLM_MAX_TOKENS) -> str:
    """Unified, retry-decorated LLM helper method using LiteLLM."""
    
    response = completion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format
    )
    return response.choices[0].message.content