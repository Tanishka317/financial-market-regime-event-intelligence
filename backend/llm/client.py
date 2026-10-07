"""
OpenAI LLM Client & Synthesis Layer
Financial Market Regime & Event Intelligence Engine

Manages interactions with OpenAI Chat Completion models (default: gpt-4o-mini),
enforces API key safety, model configuration, timeouts, and error handling.
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from backend.llm.prompts import SYSTEM_PROMPT

# Ensure environment variables are loaded from .env
load_dotenv()

logger = logging.getLogger("llm.client")


class OpenAILLMClient:
    """
    OpenAI Chat Completion API Client wrapper for grounded financial intelligence synthesis.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.model = model or os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout = timeout

    def is_available(self) -> bool:
        """
        Returns True if LLM synthesis is explicitly enabled and a valid API key is present.
        """
        enabled = os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() in ("true", "1", "yes")
        if not enabled:
            logger.info("LLM synthesis disabled via ENABLE_LLM_SYNTHESIS environment flag.")
            return False

        if not self._api_key or self._api_key.strip() in ("", "your_openai_api_key_here"):
            logger.info("LLM synthesis unavailable: OPENAI_API_KEY is missing or unconfigured.")
            return False

        return True

    def generate_synthesis(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Invokes OpenAI Chat Completion API to generate a grounded synthesis response.

        Accepts:
            prompt: User/context prompt string
            system_prompt: System prompt string (defaults to SYSTEM_PROMPT)
            temperature: Sampling temperature (default: 0.2 for analytical deterministic grounding)

        Returns:
            Grounded synthesis string response.

        Raises:
            ValueError: If API key is missing or synthesis is disabled.
            RuntimeError: If OpenAI API request fails or times out.
        """
        if not self.is_available():
            raise ValueError(
                "OPENAI_API_KEY environment variable is missing or LLM synthesis is disabled. "
                "Unable to call OpenAI Chat Completions API."
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key.strip(), timeout=self.timeout)

            messages = [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"OpenAI Chat Completion failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"OpenAI API invocation failed: {e}")
