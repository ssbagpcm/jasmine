from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Theme:
    name: str = "dark"
    prompt_name: str = "#e5e7eb bold"
    prompt_path: str = "#86efac bold"
    prompt_arrow: str = "#a3a3a3 bold"
    completion_bg: str = "bg:#252525 #d1d5db"
    completion_current: str = "bg:#22c55e #ffffff bold"
    completion_meta: str = "#9ca3af"
    scrollbar_bg: str = "bg:#171717"
    scrollbar_button: str = "bg:#525252"
    user_text: str = "#e5e7eb"
    assistant_text: str = "#d6d6d6"


DARK_THEME = Theme()
LIGHT_THEME = Theme(
    name="light",
    prompt_name="#1f2937 bold",
    prompt_path="#166534 bold",
    prompt_arrow="#6b7280 bold",
    completion_bg="bg:#f1f5f9 #1f2937",
    completion_current="bg:#16a34a #ffffff bold",
    completion_meta="#6b7280",
    scrollbar_bg="bg:#e2e8f0",
    scrollbar_button="bg:#94a3b8",
    user_text="#1f2937",
    assistant_text="#333333",
)


@dataclass
class JasmineConfig:
    model: str | None = None
    reasoning_effort: str | None = None
    thinking: str | None = None
    trusted_prefixes: list[str] = field(default_factory=list)
    theme: str = "dark"

    @classmethod
    def load(cls, root: Path) -> JasmineConfig:
        config_path = root / ".jasmine.toml"
        if not config_path.exists():
            return cls()

        cfg = cls()
        try:
            raw = config_path.read_text(encoding="utf-8")
            parsed = cls._parse_toml(raw)
        except Exception:
            return cfg

        cfg.model = parsed.get("model")
        cfg.reasoning_effort = parsed.get("reasoning_effort")
        cfg.thinking = parsed.get("thinking")
        cfg.theme = parsed.get("theme", "dark")
        trusted = parsed.get("trusted_prefix") or parsed.get("trusted_prefixes")
        if trusted:
            if isinstance(trusted, list):
                cfg.trusted_prefixes = [str(item) for item in trusted if str(item).strip()]
            else:
                cfg.trusted_prefixes = [str(trusted).strip()]
        return cfg

    @staticmethod
    def _parse_toml(raw: str) -> dict[str, Any]:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(raw)


def get_theme(config: JasmineConfig) -> Theme:
    if config.theme == "light":
        return LIGHT_THEME
    env_theme = os.environ.get("JASMINE_THEME", "").lower()
    if env_theme == "light":
        return LIGHT_THEME
    return DARK_THEME
