from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from .ai import AIEvent


class OpenAIBackend:
    """Generic OpenAI-compatible backend.

    Works with OpenAI, Anthropic (via compatible proxy), Groq, Together, local
    Ollama/vLLM servers, or any API that speaks the OpenAI chat completions
    protocol.

    Set JASMINE_API_KEY and JASMINE_BASE_URL in the environment, or pass them
    directly.  Falls back to OPENAI_API_KEY / OPENAI_BASE_URL when Jasmine-
    specific vars are not set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
        thinking_default: bool = False,
        supports_thinking: bool = False,
        supports_tool_choice: bool = True,
        supports_vision: bool = True,
        provider_name: str = "openai",
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
        cached_input_price_per_million: float | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI backend needs the OpenAI Python SDK. Install with: pip install -e ."
            ) from exc

        self.api_key = (
            api_key
            or os.environ.get("JASMINE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("AI_CODE_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set JASMINE_API_KEY, OPENAI_API_KEY, or AI_CODE_API_KEY."
            )

        self.base_url = (
            base_url
            or os.environ.get("JASMINE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("AI_CODE_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model or os.environ.get("JASMINE_MODEL") or os.environ.get("AI_CODE_MODEL", "gpt-4o")
        stream_env = os.environ.get("JASMINE_STREAM") or os.environ.get("AI_CODE_STREAM", "1")
        self.stream_enabled = stream if stream is not None else stream_env != "0"
        self.reasoning_effort = reasoning_effort or os.environ.get("JASMINE_REASONING_EFFORT") or os.environ.get("AI_CODE_REASONING_EFFORT") or None
        if enable_thinking is None:
            thinking_env = os.environ.get("JASMINE_THINKING")
            if thinking_env is None:
                thinking_env = os.environ.get("AI_CODE_THINKING")
            enable_thinking = thinking_default if thinking_env is None else str(thinking_env).strip().lower() not in {"0", "off", "false", "no", "disabled"}
        self.enable_thinking = enable_thinking
        self.supports_thinking = supports_thinking
        self.supports_tool_choice = supports_tool_choice
        self.supports_vision = supports_vision
        self.provider_name = provider_name
        self.pricing = {
            "input_price_per_million": input_price_per_million,
            "output_price_per_million": output_price_per_million,
            "cached_input_price_per_million": cached_input_price_per_million,
        }
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[AIEvent]:
        provider_messages = self._normalize_messages(messages)
        provider_tools = self._normalize_tools(tools)
        try:
            if self.stream_enabled:
                async for event in self._streaming_completion(provider_messages, provider_tools):
                    yield event
            else:
                async for event in self._single_completion(provider_messages, provider_tools):
                    yield event
            yield {"type": "done"}
        except Exception as exc:
            detail = str(exc) or repr(exc)
            yield {"type": "error", "error": f"{type(exc).__qualname__}: {detail}"}
            yield {"type": "done"}

    def _completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            if self.supports_tool_choice:
                kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if self.reasoning_effort and (not self.supports_thinking or self.enable_thinking):
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.supports_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if self.enable_thinking else "disabled"}}
        return kwargs

    async def _single_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[AIEvent]:
        kwargs = self._completion_kwargs(messages, tools, stream=False)
        response = await self.client.chat.completions.create(**kwargs)
        if not getattr(response, "choices", None):
            return
        choice = response.choices[0]
        message = choice.message

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            yield {"type": "thinking", "content": str(reasoning)}
        content = getattr(message, "content", None)
        if content:
            yield {"type": "text", "content": str(content)}

        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "") if function else ""
            arguments = getattr(function, "arguments", "{}") if function else "{}"
            yield {"type": "tool_call", "id": str(getattr(call, "id", "") or ""), "name": name, "args": self._json_args(arguments)}
        usage = self._usage_dict(getattr(response, "usage", None))
        if usage:
            yield {"type": "usage", "usage": usage}

    async def _streaming_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[AIEvent]:
        kwargs = self._completion_kwargs(messages, tools, stream=True)
        stream = await self.client.chat.completions.create(**kwargs)
        tool_buffers: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            usage = self._usage_dict(getattr(chunk, "usage", None))
            if usage:
                yield {"type": "usage", "usage": usage}
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None)
            if reasoning:
                yield {"type": "thinking", "content": str(reasoning)}
            content = getattr(delta, "content", None)
            if content:
                yield {"type": "text", "content": str(content)}

            for call in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(call, "index", 0) or 0)
                buffer = tool_buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
                call_id = getattr(call, "id", None)
                if call_id:
                    buffer["id"] = str(call_id)
                function = getattr(call, "function", None)
                if function:
                    name = getattr(function, "name", None)
                    args = getattr(function, "arguments", None)
                    if name:
                        buffer["name"] += str(name)
                    if args:
                        buffer["arguments"] += str(args)

        for index in sorted(tool_buffers):
            item = tool_buffers[index]
            name = item.get("name", "").strip()
            if not name:
                continue
            yield {"type": "tool_call", "id": item.get("id", ""), "name": name, "args": self._json_args(item.get("arguments") or "{}")}

    def _normalize_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                normalized.append(tool)
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool.get("name", "")),
                        "description": str(tool.get("description", "")),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        return [tool for tool in normalized if tool["function"]["name"]]

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if content is None:
                content = ""
            if isinstance(content, list):
                pass  # multimodal content — pass through as-is
            else:
                content = str(content)
            if role in {"system", "user"}:
                normalized.append({"role": role, "content": content})
                continue
            if role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": content}
                if "reasoning_content" in message:
                    item["reasoning_content"] = str(message.get("reasoning_content") or "")
                if message.get("tool_calls"):
                    item["tool_calls"] = message["tool_calls"]
                normalized.append(item)
                continue
            if role == "tool" and message.get("tool_call_id"):
                normalized.append({"role": "tool", "tool_call_id": str(message["tool_call_id"]), "content": content})
                continue
            if role == "tool":
                normalized.append({"role": "tool", "tool_call_id": "unknown", "content": content})
                continue
            normalized.append({"role": "user", "content": content})
        return normalized

    def _usage_dict(self, usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        keys = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "completion_tokens_details",
        )
        return {key: getattr(usage, key) for key in keys if getattr(usage, key, None) is not None}

    def _json_args(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw or "{}").strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {"value": value}
        except json.JSONDecodeError:
            return {"raw": text}
