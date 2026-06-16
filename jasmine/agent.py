from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

from ._markdown import CompactMarkdown
from .ai import AIBackend
from .prompts import SYSTEM_PROMPT
from .tools import ToolRegistry
from .ui import TerminalUI, _format_duration


class Agent:
    TOOL_ALIASES = {
        "shell": "exec_command",
        "bash": "exec_command",
        "terminal": "exec_command",
        "run_command": "exec_command",
        "stdin": "write_stdin",
        "send_stdin": "write_stdin",
        "patch": "apply_patch",
        "explore": "multi_tool_use_parallel",
        "parallel": "multi_tool_use_parallel",
        "batch": "multi_tool_use_parallel",
        "search": "web_search",
        "web": "web_search",
        "web_search": "web_search",
        "duckduckgo": "web_search",
        "fetch": "web_extract",
        "read_url": "web_extract",
        "web_fetch": "web_extract",
        "extract_url": "web_extract",
    }

    def __init__(self, backend: AIBackend, tools: ToolRegistry, ui: TerminalUI) -> None:
        self.backend = backend
        self.tools = tools
        self.ui = ui
        self.root = tools.root
        self.messages: list[dict[str, Any]] = []
        self.max_model_rounds = 256
        self.max_same_tool_calls = 2
        self._turn_counter = 0
        self._tool_result_cache: dict[str, dict[str, Any]] = {}
        self._tool_call_counts: dict[str, int] = {}
        self._missing_shell_commands: set[str] = set()
        self._narration_continuations = 0
        self.changed_files: set[str] = set()
        self._cached_system_prompt: str | None = None
        self._cached_changed_files_sig: str = ""
        self.provider_name = str(getattr(backend, "provider_name", backend.__class__.__name__))
        self.pricing = dict(getattr(backend, "pricing", {}) or {})
        self.usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
        }
        self._reset_messages()
        self._load_usage()

    def set_workspace(self, tools: ToolRegistry) -> None:
        self.tools = tools
        self.root = tools.root
        self.changed_files = set()
        self._tool_result_cache = {}
        self._tool_call_counts = {}
        self._cached_system_prompt = None
        self._reset_messages()

    def set_backend(self, backend: AIBackend) -> None:
        self.backend = backend
        self.provider_name = str(getattr(backend, "provider_name", backend.__class__.__name__))
        self.pricing = dict(getattr(backend, "pricing", {}) or {})
        self._cached_system_prompt = None

    def _reset_messages(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt()}]

    def _strip_orphaned_tool_calls(self) -> None:
        """Remove tool_calls from the last assistant message if its tool responses
        are missing.  Called after a user interruption to keep history valid."""
        if not self.messages:
            return
        last = self.messages[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            last.pop("tool_calls", None)

    def _system_prompt(self) -> str:
        # Use cached prompt unless changed_files was modified
        sig = ",".join(sorted(self.changed_files))
        if self._cached_system_prompt is not None and self._cached_changed_files_sig == sig:
            return self._cached_system_prompt
        self._cached_changed_files_sig = sig

        agents_md = self._load_agents_md()

        tool_names = ", ".join(tool["name"] for tool in self._model_tool_schemas())
        vision = "available" if bool(getattr(self.backend, "supports_vision", True)) else "text-only"
        thinking = "available" if bool(getattr(self.backend, "supports_thinking", False)) else "standard"
        changed = ", ".join(sorted(self.changed_files)[-12:]) if self.changed_files else "(none)"
        runtime = (
            "\n\n## Runtime Context\n"
            f"- workspace root: {self.root}\n"
            f"- provider: {self.provider_name}\n"
            f"- provider vision: {vision}\n"
            f"- provider reasoning: {thinking}\n"
            f"- available tools: {tool_names}\n"
            f"- files changed this session: {changed}\n"
            "- Use this live context. Do not rediscover provider capabilities or tool names with shell scripts."
        )

        # Combine : SYSTEM_PROMPT + AGENTS.md + runtime
        base_prompt = SYSTEM_PROMPT
        if agents_md:
            base_prompt = f"{base_prompt}\n\n---\n\n{agents_md}"
        prompt = base_prompt + runtime
        self._cached_system_prompt = prompt
        return prompt

    def _load_agents_md(self) -> str:
        """Load AGENTS.md from <workspace>/.jasmine/AGENTS.md"""
        agents_path = self.root / ".jasmine" / "AGENTS.md"
        if agents_path.exists():
            try:
                return agents_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return ""

    def _model_tool_schemas(self) -> list[dict[str, Any]]:
        return self.tools.model_schemas()

    def show_usage(self) -> None:
        self.ui.print_usage({**self.usage, "provider": self.provider_name})

    @property
    def _usage_path(self) -> Path:
        return self.root / ".jasmine" / "usage.json"

    def _load_usage(self) -> None:
        try:
            data = json.loads(self._usage_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in self.usage:
                    if key in data:
                        self.usage[key] = data[key]
        except Exception:
            pass

    def _save_usage(self) -> None:
        try:
            self._usage_path.parent.mkdir(parents=True, exist_ok=True)
            self._usage_path.write_text(json.dumps(self.usage, indent=2), encoding="utf-8")
        except Exception:
            pass

    def clear_context(self, *, announce: bool = True) -> None:
        self._reset_messages()
        self.ui.plan_items = []
        if announce:
            self.ui.console.print("[green]•[/] [#9e9e9e]context cleared[/#9e9e9e]")

    def load_messages(self, messages: list[dict[str, Any]]) -> None:
        """Replace the current conversation with a previously saved one."""
        self.messages = [{"role": "system", "content": self._system_prompt()}] + [
            m for m in messages if m.get("role") != "system"
        ]
        self.changed_files = set()
        self._tool_result_cache = {}
        self._tool_call_counts = {}
        self._narration_continuations = 0

    async def run_user_turn(self, user_text: str) -> None:
        self._turn_counter += 1
        self._tool_result_cache = {}
        self._tool_call_counts = {}
        self._narration_continuations = 0
        self.messages[0] = {"role": "system", "content": self._system_prompt()}
        self.messages.append({"role": "user", "content": user_text})
        start_time = time.time()
        try:
            await self._run_ai_loop()
        except asyncio.CancelledError:
            # If the last assistant message has pending tool_calls with no matching
            # tool responses, strip the tool_calls so the conversation remains valid.
            self._strip_orphaned_tool_calls()
            self.messages.append({"role": "user", "content": "[previous run interrupted by user]"})
            raise
        else:
            elapsed = time.time() - start_time
            if elapsed >= 45:
                self.ui.console.print(f"[#6b6b6b italic]worked for {_format_duration(elapsed)}[/#6b6b6b italic]")
            self.ui.console.print()

    async def _run_ai_loop(self) -> None:
        for round_index in range(1, self.max_model_rounds + 1):
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            async with self.ui.assistant_stream() as stream:
                self.usage["requests"] += 1
                async for event in self.backend.stream(self.messages, self._model_tool_schemas()):  # type: ignore[attr-defined]
                    event_type = event.get("type")
                    if event_type == "thinking":
                        chunk = str(event.get("content", ""))
                        thinking_parts.append(chunk)
                        stream.append_thinking(chunk)
                    elif event_type == "text":
                        chunk = str(event.get("content", ""))
                        text_parts.append(chunk)
                        stream.append(chunk)
                    elif event_type == "tool_call":
                        ordinal = len(tool_calls) + 1
                        tool_calls.append(
                            {
                                "id": str(event.get("id") or f"jasmine_{self._turn_counter}_{round_index}_{ordinal}"),
                                "name": str(event.get("name") or ""),
                                "args": self._coerce_args(event.get("args") or {}),
                            }
                        )
                    elif event_type == "usage":
                        self._record_usage(event.get("usage"))
                    elif event_type == "error":
                        message = f"\n**AI error:** `{event.get('error')}`"
                        text_parts.append(message)
                        stream.append(message)
                stream.set_followed_by_tools(bool(tool_calls))

            assistant_text = "".join(text_parts).strip()
            if tool_calls:
                self.messages.append(self._assistant_tool_message(assistant_text, tool_calls, "".join(thinking_parts)))
                if not await self._run_tool_calls(tool_calls):
                    return
                continue
            if assistant_text:
                self.messages.append({"role": "assistant", "content": assistant_text})
            if self._should_nudge_narration(assistant_text):
                self._narration_continuations += 1
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue now with the next concrete tool call. "
                            "If the task is already complete, give the final answer instead."
                        ),
                    }
                )
                continue
            return

        self.ui.console.print(f"[yellow]•[/] [#9e9e9e]stopped after {self.max_model_rounds} model rounds to prevent a runaway loop[/#9e9e9e]")

    def _assistant_tool_message(self, content: str, tool_calls: list[dict[str, Any]], reasoning_content: str = "") -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": [
                {
                    "id": item["id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": json.dumps(item["args"], ensure_ascii=False),
                    },
                }
                for item in tool_calls
            ],
        }

    async def _run_tool_calls(self, tool_calls: list[dict[str, Any]]) -> bool:
        for item in tool_calls:
            name, args = self._normalize_tool_call(item["name"], item["args"])
            signature = self._tool_signature(name, args)
            self._tool_call_counts[signature] = self._tool_call_counts.get(signature, 0) + 1
            if self._tool_call_counts[signature] > self.max_same_tool_calls:
                result = {
                    "ok": False,
                    "error": "Identical tool call repeated too many times in this turn. Use existing evidence or change approach.",
                }
            else:
                invalid = self._validate_tool_call(name, args)
                if invalid is not None:
                    result = invalid
                else:
                    approval = await self._approve_tool_if_needed(name, args)
                    if approval is not None:
                        result = approval
                    else:
                        cached = self._tool_result_cache.get(signature) if self._is_cacheable(name, args) else None
                        if cached is not None:
                            result = {
                                "ok": True,
                                "cached_by_agent": True,
                                "note": "identical read-only shell command already satisfied in this turn",
                            }
                        elif name == "ask_user":
                            result = await self._call_tool(name, args)
                        else:
                            async with self.ui.tool_activity(name, args):
                                result = await self._call_tool(name, args)
                            if self._is_cacheable(name, args) and result.get("ok") is not False:
                                self._tool_result_cache[signature] = result
            if name == "apply_patch" and result.get("changed"):
                self.changed_files.update(str(path) for path in result.get("paths", []))  # type: ignore[attr-defined]
                self._tool_result_cache = {}
                self._tool_call_counts = {}
            self._remember_missing_shell_command(name, args, result)
            self.ui.print_tool_result(name, result)
            model_result = self._tool_result_for_model(result)
            can_send_vision = bool(getattr(self.backend, "supports_vision", True))
            if name == "view_image" and result.get("base64") and not can_send_vision:
                model_result["vision_unavailable"] = True
                model_result["vision_note"] = (
                    f"Provider {self.provider_name} is configured as text-only. "
                    "Do not claim to see the image; ask the user to switch to a vision-capable provider."
                )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["id"],
                    "content": json.dumps(model_result, ensure_ascii=False),
                }
            )
            if name == "view_image" and result.get("base64") and can_send_vision:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"The image `{result.get('path', '')}` ({result.get('width')}x{result.get('height')}) is attached as a multimodal image_url block. Analyze the actual pixels directly. Describe what you see in detail: subject, scene, colors, composition, notable elements, text if any, and overall impression. Do not rely only on tool metadata."},
                            {"type": "image_url", "image_url": {"url": f"data:{result['base64_mime']};base64,{result['base64']}"}},
                        ],
                    }
                )
            if result.get("denied"):
                return False
        return True

    async def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "update_plan":
            return self._update_plan(args)
        if name == "ask_user":
            return await self._ask_user(args)
        if name == "multi_tool_use_parallel":
            return await self._run_parallel_tools(args)
        return await self.tools.call(name, args)

    async def _run_parallel_tools(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_uses = args.get("tool_uses") or []
        if not isinstance(raw_uses, list) or not 1 <= len(raw_uses) <= 4:
            return {"ok": False, "error": "tool_uses must contain 1-4 independent read-only operations"}

        prepared: list[tuple[str, dict[str, Any]]] = []
        for raw in raw_uses:
            # Accept a bare string as an exec_command shorthand
            if isinstance(raw, str):
                raw = {"recipient_name": "exec_command", "parameters": {"command": raw}}
            if not isinstance(raw, dict):
                return {"ok": False, "error": "each parallel operation must be an object or a command string"}

            # Resolve the nested tool name: try every reasonable field the model might send
            raw_name = (
                raw.get("recipient_name")
                or raw.get("name")
                or raw.get("tool_name")
                or raw.get("function")
                or raw.get("tool")
                or ""
            )
            # If the model sent a nested function object (OpenAI style), unwrap it
            if isinstance(raw_name, dict):
                raw_name = raw_name.get("name", "")
            name = str(raw_name)
            # Resolve nested args: prefer parameters, then args, then auto-wrap if the
            # model flattened command / path directly into the operation object.
            if "parameters" in raw and isinstance(raw["parameters"], dict):
                nested_args = dict(raw["parameters"])
            elif "args" in raw and isinstance(raw["args"], dict):
                nested_args = dict(raw["args"])
            else:
                # Auto-wrap: pull known fields into a parameters dict
                nested_args = {}
                for key in ("command", "cmd", "shell", "path", "cwd", "workdir", "query", "url"):
                    if key in raw:
                        nested_args[key] = raw[key]
                if not nested_args:
                    return {"ok": False, "error": f"parallel operation {name or '(missing name)'} needs parameters, args, or an inline command/path"}

            name, nested_args = self._normalize_tool_call(name, nested_args)

            if not name:
                if "command" in nested_args or "cmd" in nested_args:
                    name = "exec_command"
                elif "path" in nested_args:
                    name = "view_image"
                elif "url" in nested_args:
                    name = "web_extract"
                elif "query" in nested_args:
                    name = "web_search"
                else:
                    name = "exec_command"

            read_only_tools = {"exec_command", "view_image", "web_search", "web_extract"}
            if name not in read_only_tools:
                allowed = ", ".join(sorted(read_only_tools))
                return {"ok": False, "error": f"parallel tool not allowed: {name}. Use only read-only tools: {allowed}."}
            if name == "exec_command" and any(nested_args.get(key) in (True, "true") or (key == "session_name" and nested_args.get(key)) for key in ("background", "session_name", "foreground")):
                return {"ok": False, "error": "parallel exec_command operations must be read-only foreground captures"}
            if name == "exec_command" and not self.ui.is_safe_auto_command(str(nested_args.get("command", ""))):
                return {"ok": False, "error": "parallel exec_command operations must be recognized local read-only commands"}
            invalid = self._validate_tool_call(name, nested_args)
            if invalid is not None:
                return invalid
            approval = await self._approve_tool_if_needed(name, nested_args)
            if approval is not None:
                return approval
            prepared.append((name, nested_args))

        results = await asyncio.gather(
            *(self.tools.call(name, nested_args) for name, nested_args in prepared)
        )
        for (name, nested_args), result in zip(prepared, results, strict=True):
            self._remember_missing_shell_command(name, nested_args, result)
        return {
            "ok": all(result.get("ok") is not False for result in results),
            "results": [
                {"name": name, "args": nested_args, "result": result}
                for (name, nested_args), result in zip(prepared, results, strict=True)
            ],
        }

    def _update_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_items = args.get("plan") or args.get("items") or []
        if not isinstance(raw_items, list):
            return {"ok": False, "error": "plan must be a list"}
        items = [
            {
                "text": str(item.get("step") or item.get("text") or "").strip(),
                "status": str(item.get("status", "pending")),
            }
            for item in raw_items
            if isinstance(item, dict) and str(item.get("step") or item.get("text") or "").strip()
        ]
        if sum(item["status"] == "in_progress" for item in items) > 1:
            return {"ok": False, "error": "at most one plan item may be in_progress"}
        changed = self.ui.update_plan(items)
        return {"ok": True, "plan": items, "changed": changed}

    async def _ask_user(self, args: dict[str, Any]) -> dict[str, Any]:
        question = str(args.get("question", "")).strip()
        if not question:
            return {"ok": False, "error": "question is required"}
        raw_options = args.get("options") or []
        options = [str(o).strip() for o in raw_options if str(o).strip()] if isinstance(raw_options, list) else []

        self.ui.console.print()
        self.ui.console.print(CompactMarkdown(f"? {question}"))
        if options:
            for idx, option in enumerate(options, start=1):
                self.ui.console.print(f"  [bold #e5e7eb]{idx}.[/] [#d1d5db]{option}[/#d1d5db]")
            self.ui.console.print("[#737373]  Enter a number or type your own answer[/#737373]")
        try:
            answer = await self.ui.line_prompt("> ", style_class="")
        except (EOFError, KeyboardInterrupt):
            return {"ok": False, "denied": True, "error": "User cancelled the question", "answer": ""}
        answer = answer.strip()
        # If the answer is a number matching an option, resolve it
        if options:
            try:
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    answer = options[idx]
            except ValueError:
                pass  # user typed custom text
        return {"ok": True, "question": question, "answer": answer}

    async def _approve_tool_if_needed(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name != "exec_command":
            return None
        command = str(args.get("command", ""))
        cwd = str(args.get("cwd") or ".")
        approved, _remembered = await self.ui.approve_command(
            command,
            cwd=cwd,
            reason=str(args.get("justification") or ""),
            force=args.get("sandbox_permissions") == "require_escalated",
        )
        if approved:
            return None
        return {"ok": False, "denied": True, "error": "User denied the shell command. Stop and wait for new instructions.", "command": command}

    def _validate_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name != "exec_command":
            return None
        command = str(args.get("command", "")).strip()
        executable = self._leading_shell_executable(command)
        if executable in self._missing_shell_commands:
            return {
                "ok": False,
                "error": f"`{executable}` is unavailable in this environment. Use an available fallback instead of retrying it.",
                "command": command,
            }
        try:
            parts = shlex.split(command)
        except ValueError:
            return None
        if not parts:
            return None
        exe = Path(parts[0]).name
        if exe in {"cat", "head", "tail"}:
            targets = [part for part in parts[1:] if not part.startswith("-")]
            if not targets:
                if args.get("background") or args.get("session_name"):
                    return None
                return {"ok": False, "error": f"Do not run unbounded {exe}. Use sed -n '<start>,<end>p' for a focused range."}
        return None

    def _leading_shell_executable(self, command: str) -> str:
        try:
            parts = shlex.split(command)
        except ValueError:
            return ""
        return Path(parts[0]).name if parts else ""

    def _remember_missing_shell_command(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        if name != "exec_command" or result.get("returncode") != 127:
            return
        output = str(result.get("output", ""))
        if "command not found" not in output:
            return
        executable = self._leading_shell_executable(str(args.get("command", "")))
        if not executable:
            return
        self._missing_shell_commands.add(executable)
        result["hint"] = f"`{executable}` is unavailable. Use an available fallback and do not retry it."

    def _normalize_tool_call(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        normalized = self.TOOL_ALIASES.get(name, name)
        next_args = dict(args)
        if normalized == "exec_command" and "command" not in next_args:
            for key in ("cmd", "shell", "value", "raw"):
                if key in next_args:
                    next_args["command"] = next_args[key]
                    break
        if normalized == "exec_command":
            if "cwd" not in next_args and "workdir" in next_args:
                next_args["cwd"] = next_args["workdir"]
            if "foreground" not in next_args and "tty" in next_args:
                next_args["foreground"] = next_args["tty"]
        if normalized == "apply_patch" and "patch" not in next_args:
            for key in ("diff", "value", "raw"):
                if key in next_args:
                    next_args["patch"] = next_args[key]
                    break
        if normalized == "apply_patch":
            aliases = {
                "file": "path",
                "filename": "path",
                "old": "old_text",
                "before": "old_text",
                "new": "new_text",
                "after": "new_text",
            }
            for source, target in aliases.items():
                if target not in next_args and source in next_args:
                    next_args[target] = next_args[source]
        if normalized == "write_stdin" and "chars" not in next_args:
            for key in ("input", "text", "value", "raw"):
                if key in next_args:
                    next_args["chars"] = next_args[key]
                    break
        return normalized, next_args

    def _coerce_args(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            value = json.loads(str(raw))
            return value if isinstance(value, dict) else {"value": value}
        except Exception:
            return {"raw": str(raw)}

    def _tool_signature(self, name: str, args: dict[str, Any]) -> str:
        raw = json.dumps([name, args], sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_cacheable(self, name: str, args: dict[str, Any]) -> bool:
        if name != "exec_command" or args.get("background") or args.get("session_name") or args.get("foreground"):
            return False
        command = str(args.get("command", "")).strip()
        return self.ui.is_safe_auto_command(command)

    def _should_nudge_narration(self, text: str) -> bool:
        if self._narration_continuations >= 1 or not text or len(text) > 500:
            return False
        lowered = text.lower()
        if any(word in lowered for word in ("done", "complete", "fixed", "terminé", "corrigé", "bloqué", "?")):
            return False
        if any(word in lowered for word in ("exec_command", "apply_patch", "update_plan", "write_stdin")):
            return False
        return any(
            re.search(pattern, lowered)
            for pattern in (
                r"\blet me\b",
                r"\bi will\b",
                r"\bi'll\b",
                r"\bje vais\b",
                r"\bje commence\b",
                r"\bon va\b",
            )
        )

    def _record_usage(self, raw_usage: Any) -> None:
        if not isinstance(raw_usage, dict):
            return
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ):
            self.usage[key] += int(raw_usage.get(key, 0) or 0)
        details = raw_usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            self.usage["reasoning_tokens"] += int(details.get("reasoning_tokens", 0) or 0)
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
        cache_hit_tokens = int(raw_usage.get("prompt_cache_hit_tokens", 0) or 0)
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        if not cache_hit_tokens and isinstance(prompt_details, dict):
            cache_hit_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
        cache_miss_tokens = int(raw_usage.get("prompt_cache_miss_tokens", 0) or 0)
        if not cache_miss_tokens:
            cache_miss_tokens = max(0, prompt_tokens - cache_hit_tokens)
        input_price = self.pricing.get("input_price_per_million")
        output_price = self.pricing.get("output_price_per_million")
        cached_input_price = self.pricing.get("cached_input_price_per_million")
        if input_price is not None:
            self.usage["cost_usd"] += cache_miss_tokens * float(input_price) / 1_000_000
            self.usage["cost_usd"] += cache_hit_tokens * float(cached_input_price if cached_input_price is not None else input_price) / 1_000_000
        if output_price is not None:
            self.usage["cost_usd"] += completion_tokens * float(output_price) / 1_000_000

    def _tool_result_for_model(self, result: dict[str, Any]) -> dict[str, Any]:
        clean = dict(result)
        if "base64" in clean:
            clean.pop("base64")  # sent via multimodal message instead
            clean["base64"] = "<sent as multimodal image>"
        return clean
