"""LLM adapters for turbovec.llm.

Each adapter implements:

* ``complete(messages, **kwargs) -> CompletionResult`` — sync, buffered.
* ``stream(messages, **kwargs) -> Iterator[str]`` — sync, text chunks.
* ``acomplete(messages, **kwargs) -> CompletionResult`` — async, buffered.
* ``astream(messages, **kwargs) -> AsyncIterator[str]`` — async, text chunks.

Messages use the OpenAI chat format. Adapters translate to provider-native
shapes where needed.

Adapters accept ``retries=N`` to wrap the sync methods with exponential backoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Optional, Protocol, runtime_checkable

from .retries import with_retries, with_retries_async


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


class _RetryMixin:
    retries: int = 0

    def _wrap(self, fn):
        if self.retries and self.retries > 1:
            return with_retries(fn, max_attempts=self.retries)
        return fn

    def _wrap_async(self, fn):
        if self.retries and self.retries > 1:
            return with_retries_async(fn, max_attempts=self.retries)
        return fn


class OpenAIChat(_RetryMixin):
    """OpenAI chat completions. Works with OpenRouter, Together, etc. via ``base_url``."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIChat. Install with: pip install turbovec[llm-openai]"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.retries = retries

    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        return self._wrap(self._complete_raw)(messages, **kwargs)

    def _complete_raw(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
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

    def stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        kwargs.pop("stream", None)
        resp = self._client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
        for chunk in resp:
            try:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            except (IndexError, AttributeError):
                continue

    async def acomplete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        async def _call():
            resp = await self._aclient.chat.completions.create(model=self.model, messages=messages, **kwargs)
            choice = resp.choices[0].message
            usage = resp.usage
            return CompletionResult(
                text=choice.content or "",
                tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                tokens_out=getattr(usage, "completion_tokens", 0) or 0,
                model=resp.model,
                raw=resp,
            )
        return await self._wrap_async(_call)()

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        kwargs.pop("stream", None)
        resp = await self._aclient.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
        async for chunk in resp:
            try:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            except (IndexError, AttributeError):
                continue


def OpenRouterChat(model: str, *, api_key: Optional[str] = None, retries: int = 3) -> OpenAIChat:
    """OpenRouter wrapper — OpenAI-compatible endpoint."""
    return OpenAIChat(model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1", retries=retries)


class AnthropicChat(_RetryMixin):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for AnthropicChat. Install with: pip install turbovec[llm-anthropic]"
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._aclient = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        self.model = model
        self._max_tokens = max_tokens
        self.retries = retries

    def _pack(self, resp) -> CompletionResult:
        text = "".join(getattr(b, "text", "") for b in resp.content)
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=text,
            tokens_in=getattr(usage, "input_tokens", 0) or 0,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            model=resp.model,
            raw=resp,
        )

    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        return self._wrap(self._complete_raw)(messages, **kwargs)

    def _complete_raw(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        system, rest = _split_system(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        resp = self._client.messages.create(
            model=self.model, system=system or None, messages=rest, max_tokens=max_tokens, **kwargs,
        )
        return self._pack(resp)

    def stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        system, rest = _split_system(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        with self._client.messages.stream(
            model=self.model, system=system or None, messages=rest, max_tokens=max_tokens, **kwargs,
        ) as s:
            for text in s.text_stream:
                yield text

    async def acomplete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        async def _call():
            system, rest = _split_system(messages)
            max_tokens = kwargs.pop("max_tokens", self._max_tokens)
            resp = await self._aclient.messages.create(
                model=self.model, system=system or None, messages=rest, max_tokens=max_tokens, **kwargs,
            )
            return self._pack(resp)
        return await self._wrap_async(_call)()

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        system, rest = _split_system(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        async with self._aclient.messages.stream(
            model=self.model, system=system or None, messages=rest, max_tokens=max_tokens, **kwargs,
        ) as s:
            async for text in s.text_stream:
                yield text


class GeminiChat(_RetryMixin):
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: Optional[str] = None,
        retries: int = 3,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiChat. Install with: pip install turbovec[llm-gemini]"
            ) from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.retries = retries

    def _contents(self, messages: list[dict]):
        system, rest = _split_system(messages)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in rest
        ]
        return system, contents

    def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        return self._wrap(self._complete_raw)(messages, **kwargs)

    def _complete_raw(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        system, contents = self._contents(messages)
        config: dict[str, Any] = dict(kwargs.pop("config", {}) or {})
        if system:
            config["system_instruction"] = system
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=config or None, **kwargs,
        )
        usage = getattr(resp, "usage_metadata", None)
        return CompletionResult(
            text=resp.text or "",
            tokens_in=getattr(usage, "prompt_token_count", 0) or 0,
            tokens_out=getattr(usage, "candidates_token_count", 0) or 0,
            model=self.model,
            raw=resp,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        system, contents = self._contents(messages)
        config: dict[str, Any] = dict(kwargs.pop("config", {}) or {})
        if system:
            config["system_instruction"] = system
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=contents, config=config or None, **kwargs,
        ):
            if chunk.text:
                yield chunk.text

    async def acomplete(self, messages: list[dict], **kwargs: Any) -> CompletionResult:
        async def _call():
            system, contents = self._contents(messages)
            config: dict[str, Any] = dict(kwargs.pop("config", {}) or {})
            if system:
                config["system_instruction"] = system
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config or None, **kwargs,
            )
            usage = getattr(resp, "usage_metadata", None)
            return CompletionResult(
                text=resp.text or "",
                tokens_in=getattr(usage, "prompt_token_count", 0) or 0,
                tokens_out=getattr(usage, "candidates_token_count", 0) or 0,
                model=self.model,
                raw=resp,
            )
        return await self._wrap_async(_call)()

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        system, contents = self._contents(messages)
        config: dict[str, Any] = dict(kwargs.pop("config", {}) or {})
        if system:
            config["system_instruction"] = system
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config or None, **kwargs,
        ):
            if chunk.text:
                yield chunk.text


__all__ = [
    "LLM",
    "CompletionResult",
    "OpenAIChat",
    "OpenRouterChat",
    "AnthropicChat",
    "GeminiChat",
]
