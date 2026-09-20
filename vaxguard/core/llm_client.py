"""
Universal Multi-Provider LLM Client for VaxGuard.

Supports dynamic switching between providers and models at runtime.
All OpenAI-compatible providers (Groq, OpenAI, DeepSeek, Mistral, Together AI,
Ollama, vLLM) are handled via a single httpx-based client. Anthropic uses
its own message format.
"""

import os
import httpx
import json
import threading
from typing import Optional
from vaxguard.core.key_manager import GroqKeyManager
from vaxguard.core.config import DEFAULT_MODEL

# Global instance for default usage across the app
key_manager = GroqKeyManager()

# ─── Provider Registry ────────────────────────────────────────────────────────
PROVIDER_CATALOG = {
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": [
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
        ],
        "default_model": "qwen/qwen3.8-27b",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ],
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "default_model": "deepseek-chat",
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "default_model": "claude-sonnet-4-20250514",
    },
    "google": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ],
        "default_model": "gemini-2.5-flash",
    },
    "mistral": {
        "name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mixtral-8x22b",
        ],
        "default_model": "mistral-large-latest",
    },
    "together": {
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "custom": {
        "name": "Custom / Local (Ollama, vLLM, LiteLLM)",
        "base_url": "http://localhost:11434/v1",
        "env_key": "",
        "models": [],
        "default_model": "llama3.1",
    },
}


# ─── Runtime Engine State (thread-safe singleton) ─────────────────────────────
class _EngineState:
    """Thread-safe global LLM engine configuration."""

    def __init__(self):
        self._lock = threading.Lock()
        self.provider = "groq"
        self.model = DEFAULT_MODEL
        self.api_key: Optional[str] = None  # None = use env-based key
        self.custom_base_url: Optional[str] = None

    def update(self, provider: str, model: str,
               api_key: Optional[str] = None,
               custom_base_url: Optional[str] = None):
        with self._lock:
            self.provider = provider
            self.model = model
            self.api_key = api_key
            self.custom_base_url = custom_base_url

    def snapshot(self):
        with self._lock:
            return {
                "provider": self.provider,
                "model": self.model,
                "api_key": self.api_key,
                "custom_base_url": self.custom_base_url,
            }


engine_state = _EngineState()


# ─── Universal LLM Client ────────────────────────────────────────────────────

class VaxGuardLLM:
    """
    Universal LLM client that routes requests to the active provider.

    Can be instantiated with explicit overrides or will fall back to
    the global engine_state for provider/model/key selection.
    """

    def __init__(self, model: str = None, provider: str = None,
                 api_key: str = None, base_url: str = None):
        snap = engine_state.snapshot()
        self.provider = provider or snap["provider"]
        self.model = model or snap["model"]
        self._explicit_key = api_key or snap["api_key"]
        self._explicit_base_url = base_url or snap["custom_base_url"]

    def _resolve_api_key(self) -> str:
        """Resolve API key: explicit > engine_state > env var > Groq key manager."""
        if self._explicit_key:
            return self._explicit_key

        catalog = PROVIDER_CATALOG.get(self.provider, {})
        env_key = catalog.get("env_key", "")

        # For Groq, use the existing round-robin key manager
        if self.provider == "groq":
            return key_manager.get_next_key()

        # Try environment variable
        if env_key:
            val = os.getenv(env_key)
            if val:
                return val

        return ""

    def _resolve_base_url(self) -> str:
        """Resolve base URL: explicit > catalog default."""
        if self._explicit_base_url:
            return self._explicit_base_url
        catalog = PROVIDER_CATALOG.get(self.provider, {})
        return catalog.get("base_url", "https://api.openai.com/v1")

    async def generate(self, system_prompt: str, user_prompt: str,
                       max_tokens: int = 1000) -> str:
        """Route to the correct provider backend."""
        if self.provider == "anthropic":
            return await self._generate_anthropic(system_prompt, user_prompt, max_tokens)
        else:
            return await self._generate_openai_compat(system_prompt, user_prompt, max_tokens)

    async def _generate_openai_compat(self, system_prompt: str,
                                       user_prompt: str,
                                       max_tokens: int) -> str:
        """
        OpenAI-compatible chat completions endpoint.
        Works for: Groq, OpenAI, DeepSeek, Mistral, Together, Google Gemini,
                   Ollama, vLLM, LiteLLM, and any custom endpoint.
        """
        base_url = self._resolve_base_url().rstrip("/")
        api_key = self._resolve_api_key()

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _generate_anthropic(self, system_prompt: str,
                                   user_prompt: str,
                                   max_tokens: int) -> str:
        """
        Anthropic Messages API (non-OpenAI format).
        """
        base_url = self._resolve_base_url().rstrip("/")
        api_key = self._resolve_api_key()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            # Anthropic returns content as a list of blocks
            content_blocks = data.get("content", [])
            return "".join(
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            )
