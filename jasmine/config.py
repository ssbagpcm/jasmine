from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
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
class ProfileConfig:
    """Unified configuration: replaces ProviderConfig + JasmineConfig"""
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    thinking: str | None = None
    theme: str = "dark"
    supports_thinking: bool = False
    supports_vision: bool = True
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cached_input_price_per_million: float | None = None
    trusted_prefixes: list[str] = field(default_factory=list)
    git_safety: bool = False

    @classmethod
    def load(cls, profile_name: str | None = None, root: Path | None = None) -> ProfileConfig:
        """Load a profile from ~/.jasmine/profiles/<name>.toml or global config"""
        # 1. Ensure ~/.jasmine/ exists
        jasmine_home = cls._ensure_jasmine_home()

        # 2. Load the specified profile
        if profile_name:
            profile_path = jasmine_home / "profiles" / f"{profile_name}.toml"
            if profile_path.exists():
                return cls._load_profile_file(profile_path)

        # 3. Load global config (~/.jasmine/config.toml)
        config_path = jasmine_home / "config.toml"
        if config_path.exists():
            return cls._load_profile_file(config_path)

        # 4. Return an empty config with defaults
        return cls()

    @classmethod
    def _load_profile_file(cls, path: Path) -> ProfileConfig:
        """Load a TOML file into a ProfileConfig"""
        try:
            raw = path.read_text(encoding="utf-8")
            parsed = cls._parse_toml(raw)
            return cls(
                base_url=parsed.get("base_url"),
                api_key=parsed.get("api_key"),
                model=parsed.get("model"),
                reasoning_effort=parsed.get("reasoning_effort"),
                thinking=str(parsed.get("thinking", "")).lower(),
                theme=parsed.get("theme", "dark"),
                supports_thinking=parsed.get("supports_thinking", False),
                supports_vision=parsed.get("supports_vision", True),
                input_price_per_million=parsed.get("input_price_per_million"),
                output_price_per_million=parsed.get("output_price_per_million"),
                cached_input_price_per_million=parsed.get("cached_input_price_per_million"),
                trusted_prefixes=cls._parse_trusted(parsed.get("trusted_prefixes")),
                git_safety=parsed.get("git_safety", False),
            )
        except Exception:
            return cls()

    @classmethod
    def _load_workspace_config(cls, root: Path) -> ProfileConfig:
        """Load config from <workspace>/.jasmine/workspace.toml"""
        config_path = root / ".jasmine" / "workspace.toml"
        if not config_path.exists():
            return cls()
        try:
            raw = config_path.read_text(encoding="utf-8")
            parsed = cls._parse_toml(raw)
            return cls(
                trusted_prefixes=cls._parse_trusted(parsed.get("trusted_prefixes")),
                git_safety=parsed.get("git_safety", False),
            )
        except Exception:
            return cls()

    @classmethod
    def summaries(cls, root: Path | None = None) -> list[dict[str, str]]:
        """List all configured profiles with metadata for the /provider menu."""
        jasmine_home = cls._ensure_jasmine_home()
        profiles_dir = jasmine_home / "profiles"
        result: list[dict[str, str]] = []
        for path in sorted(profiles_dir.glob("*.toml")):
            name = path.stem
            try:
                cfg = cls._load_profile_file(path)
            except Exception:
                continue
            result.append({
                "name": name,
                "model": cfg.model or "(default)",
                "base_url": cfg.base_url or "(default)",
                "vision": "vision" if cfg.supports_vision else "text",
                "thinking": "thinking" if cfg.supports_thinking else "standard",
            })
        return result

    @staticmethod
    def _parse_trusted(trusted: Any) -> list[str]:
        """Parse trusted_prefixes depuis TOML (list ou string)"""
        if not trusted:
            return []
        if isinstance(trusted, list):
            return [str(item).strip() for item in trusted if str(item).strip()]
        return [str(trusted).strip()]

    @staticmethod
    def _ensure_jasmine_home() -> Path:
        """Create ~/.jasmine/ and subdirectories if missing"""
        jasmine_home = Path.home() / ".jasmine"
        jasmine_home.mkdir(parents=True, exist_ok=True)
        (jasmine_home / "profiles").mkdir(parents=True, exist_ok=True)
        return jasmine_home

    @staticmethod
    def _parse_toml(raw: str) -> dict[str, Any]:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        return tomllib.loads(raw)


def get_theme(theme_name: str) -> Theme:
    if theme_name == "light":
        return LIGHT_THEME
    env_theme = os.environ.get("JASMINE_THEME", "").lower()
    if env_theme == "light":
        return LIGHT_THEME
    return DARK_THEME
