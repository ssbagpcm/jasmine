from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

from .config import ProfileConfig

AIEvent = dict[str, Any]


class AIBackend(Protocol):
    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[AIEvent]:
        ...


class MockBackend:
    """Backend mock pour les tests offline."""

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


def load_backend(profile_name: str | None, root: Path) -> AIBackend:
    """Load a backend from a profile or use DeepSeek by default."""
    # 1. Load the profile
    profile = ProfileConfig.load(profile_name, root)

    # 2. Handle mock mode
    normalized = (profile_name or "").lower()
    if normalized in {"mock", "offline"}:
        return MockBackend()  # type: ignore[return-value]

    # 3. Default backend if no profile
    if not profile_name and not profile.base_url:
        from .deepseek_backend import DeepSeekBackend
        return DeepSeekBackend()  # type: ignore[return-value]

    # 4. Create an OpenAI-compatible backend from profile settings
    if profile.base_url:
        from .openai_backend import OpenAIBackend
        # Map profile.thinking ("on"/"off") to enable_thinking (bool)
        enable_thinking = profile.thinking == "on" if profile.thinking else False
        return OpenAIBackend(  # type: ignore[return-value]
            base_url=profile.base_url,
            api_key=profile.api_key,
            model=profile.model,
            reasoning_effort=profile.reasoning_effort,
            enable_thinking=enable_thinking,
            thinking_default=enable_thinking,
            supports_thinking=profile.supports_thinking,
            supports_vision=profile.supports_vision,
            input_price_per_million=profile.input_price_per_million,
            output_price_per_million=profile.output_price_per_million,
            cached_input_price_per_million=profile.cached_input_price_per_million,
        )

    # 5. Fallback to DeepSeek
    from .deepseek_backend import DeepSeekBackend
    return DeepSeekBackend()  # type: ignore[return-value]
