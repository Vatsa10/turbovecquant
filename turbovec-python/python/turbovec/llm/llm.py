"""LLM adapters for turbovec.llm.

Each adapter implements ``complete(messages, **kwargs) -> CompletionResult``.
Messages use the OpenAI chat format: ``[{"role": "system"|"user"|"assistant", "content": "..."}]``.
Adapters translate that to the provider's native format where needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class CompletionResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    raw: Any = None


@runtime_checkable
class LLM(Protocol):
    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult: ...


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


class OpenAIChat:
    """OpenAI chat completions. Works with OpenRouter, Together, etc., via ``base_url``."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIChat. Install with: pip install turbovec[llm-openai]"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        choice = resp.choices[0].message
        usage = resp.usage
        return CompletionResult(
            text=choice.content or "",
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            model=resp.model,
            raw=resp,
        )


def OpenRouterChat(model: str, *, api_key: Optional[str] = None) -> OpenAIChat:
    """OpenRouter wrapper — OpenAI-compatible endpoint."""
    return OpenAIChat(model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1")


class AnthropicChat:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for AnthropicChat. Install with: pip install turbovec[llm-anthropic]"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        system, rest = _split_system(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        resp = self._client.messages.create(
            model=self.model,
            system=system or None,
            messages=rest,
            max_tokens=max_tokens,
            **kwargs,
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=text,
            tokens_in=getattr(usage, "input_tokens", 0) or 0,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            model=resp.model,
            raw=resp,
        )


class GeminiChat:
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiChat. Install with: pip install turbovec[llm-gemini]"
            ) from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        system, rest = _split_system(messages)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in rest
        ]
        config: dict[str, Any] = dict(kwargs.pop("config", {}) or {})
        if system:
            config["system_instruction"] = system
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config or None,
            **kwargs,
        )
        usage = getattr(resp, "usage_metadata", None)
        return CompletionResult(
            text=resp.text or "",
            tokens_in=getattr(usage, "prompt_token_count", 0) or 0,
            tokens_out=getattr(usage, "candidates_token_count", 0) or 0,
            model=self.model,
            raw=resp,
        )


__all__ = [
    "LLM",
    "CompletionResult",
    "OpenAIChat",
    "OpenRouterChat",
    "AnthropicChat",
    "GeminiChat",
]
