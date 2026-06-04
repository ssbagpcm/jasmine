from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cached_input_price_per_million: float | None = None
    reasoning_effort: str | None = None
    supports_thinking: bool = False
    thinking_default: bool = False
    supports_tool_choice: bool = True
    supports_vision: bool = True

    @classmethod
    def load(cls, name: str, root: Path | None = None) -> ProviderConfig | None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            return None
        for path in cls.paths(name, root):
            if path.is_file():
                return cls.from_path(name, path)
        return None

    @staticmethod
    def paths(name: str, root: Path | None = None) -> list[Path]:
        paths: list[Path] = []
        if root is not None:
            current = root.resolve()
            while True:
                candidate = current / ".jasmine" / "providers" / f"{name}.toml"
                if candidate not in paths:
                    paths.append(candidate)
                parent = current.parent
                if parent == current:  # reached filesystem root
                    break
                current = parent
        home_path = Path.home() / ".jasmine" / "providers" / f"{name}.toml"
        if home_path not in paths:
            paths.append(home_path)
        return paths

    @classmethod
    def available_names(cls, root: Path | None = None) -> list[str]:
        names = {"deepseek", "mock"}
        directories = [Path.home() / ".jasmine" / "providers"]
        if root is not None:
            directories.insert(0, root.resolve() / ".jasmine" / "providers")
        for directory in directories:
            if directory.is_dir():
                names.update(path.stem for path in directory.glob("*.toml") if path.is_file())
        return sorted(names)

    @classmethod
    def summaries(cls, root: Path | None = None) -> list[dict[str, str]]:
        summaries: list[dict[str, str]] = []
        for name in cls.available_names(root):
            if name == "mock":
                summaries.append({"name": name, "model": "offline", "base_url": "local", "vision": "text-only"})
                continue
            try:
                provider = cls.load(name, root)
            except Exception as exc:
                summaries.append({"name": name, "model": "invalid config", "base_url": str(exc)})
                continue
            if provider is None:
                summaries.append({"name": name, "model": "deepseek-v4-pro", "base_url": "https://api.deepseek.com", "vision": "text-only"})
            else:
                summaries.append({
                    "name": name,
                    "model": provider.model,
                    "base_url": provider.base_url,
                    "vision": "vision" if provider.supports_vision else "text-only",
                })
        return summaries

    @classmethod
    def from_path(cls, name: str, path: Path) -> ProviderConfig:
        parsed = _parse_toml(path.read_text(encoding="utf-8"))
        api_key = str(parsed.get("api_key", "")).strip()
        api_key_env = str(parsed.get("api_key_env", "")).strip()
        if api_key_env:
            api_key = os.environ.get(api_key_env, api_key)
        base_url = str(parsed.get("base_url", "")).strip()
        model = str(parsed.get("model", "")).strip()
        if not base_url or not model or not api_key:
            raise ValueError(f"Provider {name} needs base_url, model, and api_key in {path}")
        default_supports_vision = not _is_official_deepseek(name, base_url)
        return cls(
            name=name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            input_price_per_million=_optional_float(parsed.get("input_price_per_million")),
            output_price_per_million=_optional_float(parsed.get("output_price_per_million")),
            cached_input_price_per_million=_optional_float(parsed.get("cached_input_price_per_million")),
            reasoning_effort=_optional_string(parsed.get("reasoning_effort")),
            supports_thinking=_as_bool(parsed.get("supports_thinking", False)),
            thinking_default=_as_bool(parsed.get("thinking_default", False)),
            supports_tool_choice=_as_bool(parsed.get("supports_tool_choice", True)),
            supports_vision=_as_bool(parsed.get("supports_vision", default_supports_vision)),
        )

    def create_backend(self):  # type: ignore[no-untyped-def]
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            thinking_default=self.thinking_default,
            supports_thinking=self.supports_thinking,
            supports_tool_choice=self.supports_tool_choice,
            supports_vision=self.supports_vision,
            provider_name=self.name,
            input_price_per_million=self.input_price_per_million,
            output_price_per_million=self.output_price_per_million,
            cached_input_price_per_million=self.cached_input_price_per_million,
        )


def _is_official_deepseek(name: str, base_url: str) -> bool:
    return name.lower() == "deepseek" and "api.deepseek.com" in base_url.lower()


def _parse_toml(raw: str) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    parsed = tomllib.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Provider config must be a TOML table")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}
