from __future__ import annotations

import os

from .openai_backend import OpenAIBackend


class DeepSeekBackend(OpenAIBackend):
    """OpenAI-compatible backend for the official DeepSeek API.

    A .jasmine/providers/deepseek.toml file can override these defaults.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key
            or os.environ.get("JASMINE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("AI_CODE_API_KEY"),
            base_url=base_url
            or os.environ.get("JASMINE_BASE_URL")
            or os.environ.get("AI_CODE_BASE_URL", "https://api.deepseek.com"),
            model=model
            or os.environ.get("JASMINE_MODEL")
            or os.environ.get("AI_CODE_MODEL", "deepseek-v4-pro"),
            stream=stream,
            reasoning_effort=reasoning_effort
            or os.environ.get("JASMINE_REASONING_EFFORT")
            or os.environ.get("AI_CODE_REASONING_EFFORT"),
            enable_thinking=enable_thinking,
            thinking_default=True,
            supports_thinking=True,
            supports_tool_choice=True,
            supports_vision=False,
            provider_name="deepseek",
            input_price_per_million=0.435,
            output_price_per_million=0.87,
            cached_input_price_per_million=0.003625,
        )
