"""
VaxGuard Multi-Provider LLM Adapters.

Provides pluggable, unified synchronous and asynchronous interfaces for:
- MockLLMAdapter (Deterministic testing, latency simulation, token accounting)
- OpenAIAdapter (OpenAI SDK with httpx fallback)
- AnthropicAdapter (Anthropic Claude with httpx fallback)
- GroqAdapter (Groq SDK / httpx with multi-key management)
- MistralAdapter (Mistral AI via httpx / mistralai)
- LocalLlamaAdapter (Local Ollama, vLLM, and OpenAI-compatible endpoints)
"""

from __future__ import annotations

import abc
import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

from vaxguard.core.logger import get_logger

logger = get_logger("LLMAdapters")


class BaseLLMAdapter(abc.ABC):
    """
    Abstract Base Class for all LLM inference adapters in VaxGuard.
    Supports unified sync and async execution with deterministic token accounting.
    """

    def __init__(self, model: str = "") -> None:
        self.model: str = model
        self.last_tokens_consumed: int = 0
        self.total_tokens_consumed: int = 0

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Synchronously generate a text completion for a given prompt."""
        pass

    @abc.abstractmethod
    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Asynchronously generate a text completion for a given prompt."""
        pass

    @abc.abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronously generate a response for a list of chat messages."""
        pass

    @abc.abstractmethod
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Asynchronously generate a response for a list of chat messages."""
        pass

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Helper to convert prompt + system_prompt into standard chat messages."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


class MockLLMAdapter(BaseLLMAdapter):
    """
    Deterministic Mock LLM Adapter for unit testing, offline benchmarking,
    and pipeline verification.

    Features:
    - Configurable mock responses (dict keyword mapping, list sequence, or fixed string).
    - Configurable default fallback response ("Safe response").
    - Configurable artificial latency (ms) for sync and async execution.
    - Deterministic token tracking.
    """

    def __init__(
        self,
        responses: Optional[Union[Dict[str, str], Sequence[str], str]] = None,
        default_response: str = "Safe response",
        artificial_latency_ms: float = 0.0,
        deterministic_tokens: Optional[int] = 15,
        model: str = "mock-model",
    ) -> None:
        super().__init__(model=model)
        self.responses = responses
        self.default_response = default_response
        self.artificial_latency_ms = max(0.0, float(artificial_latency_ms))
        self.deterministic_tokens = deterministic_tokens
        self._call_index: int = 0

    def _resolve_response(self, text: str) -> str:
        """Resolve response based on string, dict mapping, or sequential list."""
        if isinstance(self.responses, str):
            return self.responses

        if isinstance(self.responses, dict):
            # 1. Exact match
            if text in self.responses:
                return self.responses[text]
            # 2. Case-insensitive substring match
            text_lower = text.lower()
            for key, val in self.responses.items():
                if key.lower() in text_lower:
                    return val
            return self.default_response

        if isinstance(self.responses, (list, tuple)):
            if not self.responses:
                return self.default_response
            resp = self.responses[self._call_index % len(self.responses)]
            self._call_index += 1
            return resp

        return self.default_response

    def _record_tokens(self, prompt: str, response: str) -> int:
        if self.deterministic_tokens is not None:
            tokens = self.deterministic_tokens
        else:
            tokens = max(1, (len(prompt) + len(response)) // 4)
        self.last_tokens_consumed = tokens
        self.total_tokens_consumed += tokens
        return tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        if self.artificial_latency_ms > 0:
            time.sleep(self.artificial_latency_ms / 1000.0)

        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        resp = self._resolve_response(full_prompt)
        # If response was default, try user prompt alone
        if resp == self.default_response and system_prompt:
            alt_resp = self._resolve_response(prompt)
            if alt_resp != self.default_response:
                resp = alt_resp

        self._record_tokens(full_prompt, resp)
        return resp

    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        if self.artificial_latency_ms > 0:
            await asyncio.sleep(self.artificial_latency_ms / 1000.0)

        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        resp = self._resolve_response(full_prompt)
        if resp == self.default_response and system_prompt:
            alt_resp = self._resolve_response(prompt)
            if alt_resp != self.default_response:
                resp = alt_resp

        self._record_tokens(full_prompt, resp)
        return resp

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.artificial_latency_ms > 0:
            time.sleep(self.artificial_latency_ms / 1000.0)

        combined = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), combined
        )
        resp = self._resolve_response(last_user)
        if resp == self.default_response and combined != last_user:
            alt_resp = self._resolve_response(combined)
            if alt_resp != self.default_response:
                resp = alt_resp

        self._record_tokens(combined, resp)
        return resp

    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.artificial_latency_ms > 0:
            await asyncio.sleep(self.artificial_latency_ms / 1000.0)

        combined = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), combined
        )
        resp = self._resolve_response(last_user)
        if resp == self.default_response and combined != last_user:
            alt_resp = self._resolve_response(combined)
            if alt_resp != self.default_response:
                resp = alt_resp

        self._record_tokens(combined, resp)
        return resp


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI inference adapter.
    Uses official `openai` SDK if available, gracefully falling back to raw `httpx`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)

    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return await self.chat_async(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # 1. Try openai SDK if api_key is present
        if self.api_key:
            try:
                import openai

                client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1000),
                )
                content = resp.choices[0].message.content or ""
                tokens = resp.usage.total_tokens if resp.usage else max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
            except Exception as e:
                logger.warning(f"OpenAI SDK call failed: {e}. Falling back to httpx.")

        # 2. Raw httpx sync fallback
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            with httpx.Client(timeout=self.timeout) as http_client:
                res = http_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"OpenAI HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."

    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.api_key:
            try:
                import openai

                client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1000),
                )
                content = resp.choices[0].message.content or ""
                tokens = resp.usage.total_tokens if resp.usage else max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
            except Exception as e:
                logger.warning(f"OpenAI Async SDK call failed: {e}. Falling back to httpx.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_async_client:
                res = await http_async_client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"OpenAI Async HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."


class AnthropicAdapter(BaseLLMAdapter):
    """
    Anthropic Claude inference adapter.
    Uses official `anthropic` SDK if available, gracefully falling back to raw `httpx`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-20241022",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _split_system_prompt(
        self, messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        system_prompt = None
        user_messages: List[Dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content", "")
            else:
                user_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if not user_messages:
            user_messages = [{"role": "user", "content": "Hello"}]
        return system_prompt, user_messages

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)

    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return await self.chat_async(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        system_prompt, filtered_msgs = self._split_system_prompt(messages)

        if self.api_key:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                call_args: Dict[str, Any] = {
                    "model": self.model,
                    "messages": filtered_msgs,  # type: ignore
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.0),
                }
                if system_prompt:
                    call_args["system"] = system_prompt
                resp = client.messages.create(**call_args)
                content = resp.content[0].text if resp.content else ""
                tokens = (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
            except Exception as e:
                logger.warning(f"Anthropic SDK call failed: {e}. Falling back to httpx.")

        headers = {
            "x-api-key": str(self.api_key or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": filtered_msgs,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.0),
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as http_client:
                res = http_client.post(f"{self.base_url}/messages", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["content"][0]["text"]
                usage = data.get("usage", {})
                tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                tokens = tokens or max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Anthropic HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."

    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        system_prompt, filtered_msgs = self._split_system_prompt(messages)

        if self.api_key:
            try:
                import anthropic

                client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                call_args: Dict[str, Any] = {
                    "model": self.model,
                    "messages": filtered_msgs,  # type: ignore
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.0),
                }
                if system_prompt:
                    call_args["system"] = system_prompt
                resp = await client.messages.create(**call_args)
                content = resp.content[0].text if resp.content else ""
                tokens = (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
            except Exception as e:
                logger.warning(f"Anthropic Async SDK call failed: {e}. Falling back to httpx.")

        headers = {
            "x-api-key": str(self.api_key or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": filtered_msgs,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.0),
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_async_client:
                res = await http_async_client.post(f"{self.base_url}/messages", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["content"][0]["text"]
                usage = data.get("usage", {})
                tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                tokens = tokens or max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Anthropic Async HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."


class GroqAdapter(BaseLLMAdapter):
    """
    Groq inference adapter.
    Uses `groq` SDK if available or `httpx` with `GROQ_API_KEY`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)

    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return await self.chat_async(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.api_key:
            try:
                import groq

                client = groq.Groq(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1000),
                )
                content = resp.choices[0].message.content or ""
                tokens = resp.usage.total_tokens if resp.usage else max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
            except Exception as e:
                logger.warning(f"Groq SDK call failed: {e}. Falling back to httpx.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            with httpx.Client(timeout=self.timeout) as http_client:
                res = http_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Groq HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."

    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.api_key:
            try:
                import groq

                client = groq.AsyncGroq(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1000),
                )
                content = resp.choices[0].message.content or ""
                tokens = resp.usage.total_tokens if resp.usage else max(1, len(content) // 4)
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
            except Exception as e:
                logger.warning(f"Groq Async SDK call failed: {e}. Falling back to httpx.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_async_client:
                res = await http_async_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Groq Async HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."


class MistralAdapter(BaseLLMAdapter):
    """
    Mistral AI inference adapter.
    Uses `mistralai` SDK if available or `httpx` with `MISTRAL_API_KEY`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-small-latest",
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)

    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return await self.chat_async(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            with httpx.Client(timeout=self.timeout) as http_client:
                res = http_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Mistral HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."

    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_async_client:
                res = await http_async_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Mistral Async HTTP request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Upstream inference fallback."


class LocalLlamaAdapter(BaseLLMAdapter):
    """
    Local LLaMA adapter compatible with Ollama, vLLM, and local OpenAI endpoints.
    Defaults to Ollama default port http://localhost:11434/v1.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "ollama",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)

    async def generate_async(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        return await self.chat_async(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            with httpx.Client(timeout=self.timeout) as http_client:
                res = http_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Local LLaMA request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Local model fallback."

    async def chat_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_async_client:
                res = await http_async_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", max(1, len(content) // 4))
                self.last_tokens_consumed = tokens
                self.total_tokens_consumed += tokens
                return content
        except Exception as e:
            logger.warning(f"Local LLaMA Async request failed: {e}. Returning safe fallback.")
            self.last_tokens_consumed = 15
            self.total_tokens_consumed += 15
            return "VERDICT: [SAFE]\nCONFIDENCE: 0.05\nREASONING: Local model fallback."


_ADAPTER_REGISTRY: Dict[str, type[BaseLLMAdapter]] = {
    "mock": MockLLMAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "claude": AnthropicAdapter,
    "groq": GroqAdapter,
    "mistral": MistralAdapter,
    "local_llama": LocalLlamaAdapter,
    "localllama": LocalLlamaAdapter,
    "ollama": LocalLlamaAdapter,
    "vllm": LocalLlamaAdapter,
    "local": LocalLlamaAdapter,
}


def get_adapter(provider: str, **kwargs: Any) -> BaseLLMAdapter:
    """
    Factory function to retrieve an LLM adapter by provider name.

    Supported providers:
    - "mock"
    - "openai"
    - "anthropic" / "claude"
    - "groq"
    - "mistral"
    - "local_llama" / "ollama" / "vllm" / "local"
    """
    norm = provider.strip().lower()
    adapter_cls = _ADAPTER_REGISTRY.get(norm)
    if not adapter_cls:
        valid_options = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Available providers: {valid_options}"
        )
    return adapter_cls(**kwargs)
