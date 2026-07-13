from typing import Any, Dict, List
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential
import src.config.settings as config

def _run_completion(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    response_format: str = None,
    max_tokens: int= config.LLM_MAX_TOKENS) -> str:
    """Send one completion request through LiteLLM."""
    completion_args: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "timeout": config.LLM_TIMEOUT_SECONDS,
    }

    if model_name.startswith("ollama/"):
        completion_args["api_base"] = config.OLLAMA_API_BASE

    response = completion(**completion_args)
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _run_completion_with_retries(**kwargs: Any) -> str:
    """Retry transient hosted-provider failures."""
    return _run_completion(**kwargs)


def call_configured_llm(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    response_format: str = None,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> str:
    """Call the configured model with a bounded request time.

    Local Ollama calls use one attempt so a slow model returns an error after
    the configured timeout. Hosted providers retain retry behavior.
    """
    request_args = {
        "model_name": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "max_tokens": max_tokens,
    }

    print(f"Calling LLM '{model_name}'.")

    if model_name.startswith("ollama/"):
        return _run_completion(**request_args)

    return _run_completion_with_retries(**request_args)
