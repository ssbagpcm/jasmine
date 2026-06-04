from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, AsyncIterator, Protocol


AIEvent = dict[str, Any]


class AIBackend(Protocol):
    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[AIEvent]:
        ...


class MockBackend:
    """Offline backend for UI/tool-loop testing."""

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[AIEvent]:
        user_text = str(messages[-1].get("content", "")) if messages else ""
        lower = user_text.lower().strip()

        if lower.startswith("/read "):
            path = user_text.split(maxsplit=1)[1].strip()
            yield {"type": "tool_call", "id": "mock_read", "name": "exec_command", "args": {"command": f"sed -n '1,220p' {path}"}}
            yield {"type": "done"}
            return

        if lower.startswith("/run "):
            command = user_text.split(maxsplit=1)[1]
            yield {"type": "tool_call", "id": "mock_run", "name": "exec_command", "args": {"command": command}}
            yield {"type": "done"}
            return

        text = (
            "Mock backend ready. Streaming, full tool context, precise patches, "
            "background shells, live plans, and usage tracking are wired.\n\n"
            "Demo commands: `/read path`, `/run command`."
        )
        for char in text:
            await asyncio.sleep(0.003)
            yield {"type": "text", "content": char}
        yield {"type": "done"}


def load_backend(spec: str | None, root: Path | None = None) -> AIBackend:
    """Load the AI backend.

    Defaults to the official DeepSeek backend (OpenAI-compatible). Use --backend mock
    for offline tests, or --backend module:ClassName for your own integration.
    """
    normalized = (spec or "deepseek").lower()
    if normalized in {"mock", "offline"}:
        return MockBackend()
    from .providers import ProviderConfig

    provider = ProviderConfig.load(normalized, root)
    if provider is not None:
        return provider.create_backend()
    if normalized in {"deepseek", "real"}:
        from .deepseek_backend import DeepSeekBackend

        return DeepSeekBackend()
    assert spec is not None
    if ":" not in spec:
        if spec.lower() in {"openai", "gpt", "gpt-4", "gpt-4o"}:
            from .openai_backend import OpenAIBackend
            return OpenAIBackend()
        raise ValueError("Backend spec must match .jasmine/providers/<name>.toml, look like module:ClassName, or use 'deepseek' / 'mock'")
    if spec.startswith("openai:"):
        base_url = spec[len("openai:"):]
        if not base_url:
            base_url = os.environ.get("JASMINE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(base_url=base_url)
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()
