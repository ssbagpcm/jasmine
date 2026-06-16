from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import shlex
import signal
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .diff_utils import (
    apply_codex_update_to_text,
    apply_unified_patch_to_text,
    make_unified_diff,
    split_multi_file_diff,
)
from .terminal import InteractiveTerminal, capture_command, run_foreground_pty

ToolFunction = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    fn: ToolFunction


@dataclass
class BackgroundJob:
    job_id: int
    command: str
    cwd: Path
    started_at: float
    proc: asyncio.subprocess.Process
    session_name: str | None = None
    output_parts: list[str] = field(default_factory=list)
    reader_task: asyncio.Task[None] | None = None

    def append(self, chunk: str) -> None:
        self.output_parts.append(chunk)

    @property
    def output(self) -> str:
        return "".join(self.output_parts)


class ToolRegistry:
    """Small local tool surface exposed to the model."""

    def __init__(self, root: Path, git_safety: bool = False) -> None:
        self.root = root.resolve()
        self.tools: dict[str, Tool] = {}
        self.background_jobs: dict[int, BackgroundJob] = {}
        self.terminal_jobs: dict[int, tuple[str | None, InteractiveTerminal]] = {}
        self._next_job_id = 1
        self._undo_stack: list[list[tuple[Path, str | None]]] = []
        self.git_safety = git_safety
        self._register_builtin_tools()

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.schema}
            for tool in self.tools.values()
        ]

    def model_schemas(self) -> list[dict[str, Any]]:
        return self.schemas()

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}", "error_type": "unknown_tool"}
        try:
            return await tool.fn(args)
        except ValueError as exc:
            return {"ok": False, "error": f"ValueError: {exc}", "error_type": "invalid_input"}
        except FileNotFoundError as exc:
            return {"ok": False, "error": f"FileNotFoundError: {exc}", "error_type": "not_found"}
        except PermissionError as exc:
            return {"ok": False, "error": f"PermissionError: {exc}", "error_type": "permission_denied"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": "tool_error"}

    def _schema(self, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": required or []}

    def _register_builtin_tools(self) -> None:
        self.register(
            Tool(
                "exec_command",
                "Run a shell command in the workspace for focused inspection, search, tests, builds, or background sessions. Prefer rg, sed -n, git status/diff, and narrow checks. Set background=true for a long-running process. Combine background=true with tty=true for an interactive PTY that write_stdin and terminal_screen can control.",
                self._schema(
                    {
                        "cmd": {"type": "string"},
                        "workdir": {"type": "string", "default": "."},
                        "timeout": {"type": "integer", "default": 120},
                        "yield_time_ms": {"type": "integer", "default": 300},
                        "background": {"type": "boolean", "default": False},
                        "session_name": {"type": "string", "default": ""},
                        "tty": {"type": "boolean", "default": False},
                        "sandbox_permissions": {"type": "string", "enum": ["use_default", "require_escalated"]},
                        "justification": {"type": "string"},
                        "prefix_rule": {"type": "array", "items": {"type": "string"}},
                    },
                    ["cmd"],
                ),
                self.exec_command,
            )
        )
        self.register(
            Tool(
                "write_stdin",
                "Send characters to a background command started by exec_command. Interactive PTYs accept text and keyboard tokens such as <enter>, <up>, <ctrl+c>, <alt+x>, <shift+left>, <f5>, <down*10>, and <click 3 10>.",
                self._schema(
                    {
                        "session_id": {"type": "integer"},
                        "session_name": {"type": "string"},
                        "chars": {"type": "string"},
                        "yield_time_ms": {"type": "integer", "default": 300},
                    },
                    ["chars"],
                ),
                self.write_stdin,
            )
        )
        self.register(
            Tool(
                "terminal_screen",
                "Inspect an interactive background PTY screen, cursor, detected app, process return code, and change state. Optionally wait for the screen to change.",
                self._schema(
                    {
                        "session_id": {"type": "integer"},
                        "session_name": {"type": "string"},
                        "wait_seconds": {"type": "number", "default": 0},
                        "rows": {"type": "integer"},
                        "cols": {"type": "integer"},
                    }
                ),
                self.terminal_screen,
            )
        )
        self.register(
            Tool(
                "terminal_close",
                "Close one interactive background PTY and its child processes.",
                self._schema(
                    {
                        "session_id": {"type": "integer"},
                        "session_name": {"type": "string"},
                    }
                ),
                self.terminal_close,
            )
        )
        self.register(
            Tool(
                "apply_patch",
                "Edit workspace files. For a normal single-file change, pass path, old_text, and new_text. To create a file, pass path and new_text. To delete a file, pass path and delete=true. Use patch only for advanced multi-file or move operations.",
                self._schema(
                    {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "delete": {"type": "boolean", "default": False},
                        "patch": {"type": "string"},
                    }
                ),
                self.apply_patch,
            )
        )
        self.register(
            Tool(
                "view_image",
                "Inspect a local image path, workspace-relative or absolute. Returns dimensions, format, lightweight image stats, and a base64-encoded thumbnail that the agent sends to vision-capable providers.",
                self._schema({"path": {"type": "string"}}, ["path"]),
                self.view_image,
            )
        )
        self.register(
            Tool(
                "list_skills",
                "List available workspace and built-in skills. Use before specialized work when a reusable workflow may help.",
                self._schema({}),
                self.list_skills,
            )
        )
        self.register(
            Tool(
                "read_skill",
                "Read one available skill by name. Use only when the skill matches the current task.",
                self._schema({"name": {"type": "string"}}, ["name"]),
                self.read_skill,
            )
        )

    def _resolve(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        candidate = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return candidate

    def _resolve_image_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        candidate = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        in_workspace = candidate == self.root or self.root in candidate.parents
        if in_workspace:
            return candidate
        # Allow images outside the workspace (e.g. downloaded screenshots)
        # but verify the MIME type to prevent accessing non-image files.
        mime = mimetypes.guess_type(str(candidate))[0] or ""
        if not mime.startswith("image/"):
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return candidate

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def _is_git_repo(self) -> bool:
        """Check if the workspace root is inside a git repository."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _git_stash_safety(self) -> None:
        """Create a git stash as a safety net before file modifications."""
        try:
            import subprocess
            stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            result = subprocess.run(
                ["git", "stash", "push", "--include-untracked", "-m", f"jasmine: auto-stash before patch ({stamp})"],
                cwd=str(self.root),
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                import sys
                sys.stderr.write(f"[jasmine] git stash failed: {result.stderr.decode(errors='replace').strip() or 'unknown error'}\n")
        except Exception as exc:
            import sys
            sys.stderr.write(f"[jasmine] git stash unavailable: {type(exc).__name__}: {exc}\n")

    def _display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved == self.root or self.root in resolved.parents:
            return str(resolved.relative_to(self.root))
        return str(resolved)

    def _skill_roots(self) -> list[Path]:
        return [
            self.root / ".jasmine" / "skills",
            self.root / "skills",
            Path(__file__).resolve().parent / "skills",
        ]

    def _skill_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for root in self._skill_roots():
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                paths.setdefault(path.parent.name, path)
        return paths

    def _skill_display_path(self, path: Path) -> str:
        try:
            return self._relative(path)
        except ValueError:
            return f"builtin:{path.parent.name}"

    def _skill_description(self, path: Path) -> str:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line.lower().startswith("description:"):
                return line.split(":", 1)[1].strip()
        return ""

    async def list_skills(self, _args: dict[str, Any]) -> dict[str, Any]:
        skills = [
            {
                "name": name,
                "path": self._skill_display_path(path),
                "description": self._skill_description(path),
            }
            for name, path in self._skill_paths().items()
        ]
        output = "\n".join(
            f"{skill['name']}: {skill['description'] or skill['path']}"
            for skill in skills
        )
        return {"ok": True, "skills": skills, "output": output}

    async def read_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            return {"ok": False, "error": "skill name must contain only letters, numbers, dots, underscores, or dashes"}
        path = self._skill_paths().get(name)
        if path is None:
            available = ", ".join(self._skill_paths()) or "(none)"
            return {"ok": False, "error": f"unknown skill: {name}", "available": available}
        return {
            "ok": True,
            "name": name,
            "path": self._skill_display_path(path),
            "content": path.read_text(encoding="utf-8", errors="replace"),
        }

    async def exec_command(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args.get("cmd") or args.get("command") or "").strip()
        if not command:
            return {"ok": False, "error": "cmd is required"}
        cwd_arg = args.get("workdir", args.get("cwd", "."))
        cwd = self._resolve(str(cwd_arg))
        if not cwd.is_dir():
            return {"ok": False, "error": f"workdir is not a directory: {cwd_arg}"}
        timeout = max(1, min(int(args.get("timeout", 120)), 600))
        if bool(args.get("background") or args.get("session_name")):
            if bool(args.get("tty")):
                return self._start_terminal(command, cwd, str(args.get("session_name", "")))
            return await self._start_background(command, cwd, str(args.get("session_name", "")))
        if bool(args.get("tty") or args.get("foreground")):
            rc = run_foreground_pty(command, cwd)
            return {"ok": rc == 0, "command": command, "cwd": self._relative(cwd), "returncode": rc, "output": f"[foreground session exited with code {rc}]"}
        result = await capture_command(command, cwd, timeout=timeout)
        payload = {
            "ok": result.returncode == 0,
            "command": command,
            "cwd": self._relative(cwd),
            "returncode": result.returncode,
            "output": result.output,
        }
        hint = self._missing_path_hint(command, result.output) if result.returncode else ""
        if hint:
            payload["hint"] = hint
        return payload

    def _missing_path_hint(self, command: str, output: str) -> str:
        if not any(marker in output for marker in ("No such file or directory", "cannot open", "can't open")):
            return ""
        try:
            parts = shlex.split(command)
        except ValueError:
            return ""
        if not parts or Path(parts[0]).name not in {"cat", "head", "tail", "sed", "wc"}:
            return ""
        suggestions: list[str] = []
        for raw in parts[1:]:
            if raw.startswith("-") or "/" in raw or not Path(raw).suffix:
                continue
            for candidate in self.root.rglob(raw):
                if not candidate.is_file():
                    continue
                relative = self._relative(candidate)
                if any(part in {".git", ".venv", "__pycache__"} for part in Path(relative).parts):
                    continue
                if relative not in suggestions:
                    suggestions.append(relative)
                if len(suggestions) >= 3:
                    break
        if not suggestions:
            return ""
        return "Possible workspace path: " + ", ".join(suggestions)

    async def _start_background(self, command: str, cwd: Path, session_name: str) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            executable=os.environ.get("SHELL", "/bin/bash"),
            start_new_session=True,
        )
        job = BackgroundJob(
            job_id=self._next_job_id,
            command=command,
            cwd=cwd,
            started_at=asyncio.get_running_loop().time(),
            proc=proc,
            session_name=session_name or None,
        )
        self._next_job_id += 1

        async def read_output() -> None:
            assert proc.stdout is not None
            try:
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    job.append(chunk.decode(errors="replace"))
                await proc.wait()
            except Exception:
                pass

        job.reader_task = asyncio.create_task(read_output())
        self.background_jobs[job.job_id] = job
        return {
            "ok": True,
            "job_id": job.job_id,
            "session_name": job.session_name,
            "command": command,
            "cwd": self._relative(cwd),
            "output": "",
        }

    def _start_terminal(self, command: str, cwd: Path, session_name: str) -> dict[str, Any]:
        terminal = InteractiveTerminal(command, cwd)
        terminal.start()
        job_id = self._next_job_id
        self._next_job_id += 1
        self.terminal_jobs[job_id] = (session_name or None, terminal)
        return {
            "ok": True,
            "terminal": True,
            "job_id": job_id,
            "session_id": job_id,
            "session_name": session_name or None,
            "command": command,
            "cwd": self._relative(cwd),
            "output": terminal.get_status(),
        }

    def resize_all_terminals(self) -> None:
        """Propagate terminal size changes to all active interactive PTY sessions."""
        import shutil
        try:
            width, height = shutil.get_terminal_size()
        except Exception:
            return
        cols = max(20, min(int(width), 400))
        rows = max(5, min(int(height), 200))
        for _session_name, terminal in self.terminal_jobs.values():
            with suppress(Exception):
                terminal.resize(rows, cols)

    async def close_background_jobs(self) -> None:
        """Stop active shell sessions without leaving children or pipe readers."""
        # 1. Cancel reader tasks first so they don't hang on blocked reads
        for job in self.background_jobs.values():
            if job.reader_task is not None and not job.reader_task.done():
                job.reader_task.cancel()
        # 2. Close stdin and terminate processes
        for job in self.background_jobs.values():
            if job.proc.stdin is not None:
                job.proc.stdin.close()
            if job.proc.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(job.proc.pid, signal.SIGTERM)
        # 3. Wait for processes and reader tasks
        for job in self.background_jobs.values():
            try:
                await asyncio.wait_for(job.proc.wait(), timeout=1)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(job.proc.pid, signal.SIGKILL)
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(job.proc.wait(), timeout=1)
            if job.reader_task is not None:
                with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(job.reader_task, timeout=1)
        for _session_name, terminal in self.terminal_jobs.values():
            terminal.close()

    def _job_summary(self, job: BackgroundJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "command": job.command,
            "session_name": job.session_name,
            "running": job.proc.returncode is None,
            "returncode": job.proc.returncode,
        }

    async def write_stdin(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = int(args.get("session_id", 0) or 0)
        session_name = str(args.get("session_name", ""))
        terminal_entry = self._find_terminal(session_id, session_name)
        if terminal_entry is not None:
            job_id, saved_name, terminal = terminal_entry
            terminal.send(str(args.get("chars", "")), wait=max(0, int(args.get("yield_time_ms", 300))) / 1000)
            return {
                "ok": True,
                "terminal": True,
                "session_id": job_id,
                "session_name": saved_name,
                "output": terminal.get_status(),
            }
        job = self.background_jobs.get(session_id) if session_id else None
        if job is None and session_name:
            job = next(
                (item for item in self.background_jobs.values() if item.session_name == session_name and item.proc.returncode is None),
                None,
            )
        # Fallback: partial name match (must be exact prefix, must be unambiguous)
        if job is None and session_name:
            candidates = [
                item for item in self.background_jobs.values()
                if item.session_name and item.session_name.startswith(session_name) and item.proc.returncode is None
            ]
            if len(candidates) == 1:
                job = candidates[0]
        if job is None or job.proc.returncode is not None:
            running = [j for j in self.background_jobs.values() if j.proc.returncode is None]
            hint = ""
            if running:
                names = ", ".join(f"{j.session_name or 'job '+str(j.job_id)}" for j in running)
                hint = f" | running sessions: {names}"
            return {"ok": False, "error": "background job not found or already exited" + hint}
        if job.proc.stdin is None:
            return {"ok": False, "error": "stdin not available", "job": self._job_summary(job)}
        job.proc.stdin.write(str(args.get("chars", "")).encode())
        await job.proc.stdin.drain()
        await asyncio.sleep(max(0, int(args.get("yield_time_ms", 300))) / 1000)
        return {
            "ok": True,
            "session_id": job.job_id,
            "session_name": job.session_name,
            "job": self._job_summary(job),
            "output": job.output,
        }

    def _find_terminal(self, session_id: int, session_name: str) -> tuple[int, str | None, InteractiveTerminal] | None:
        if session_id and session_id in self.terminal_jobs:
            saved_name, terminal = self.terminal_jobs[session_id]
            return session_id, saved_name, terminal
        if session_name:
            # Exact match first
            for job_id, (saved_name, terminal) in self.terminal_jobs.items():
                if saved_name == session_name:
                    return job_id, saved_name, terminal
            # Prefix match fallback: only if unambiguous
            candidates = [
                (job_id, saved_name, terminal)
                for job_id, (saved_name, terminal) in self.terminal_jobs.items()
                if saved_name and saved_name.startswith(session_name)
            ]
            if len(candidates) == 1:
                return candidates[0]
        return None

    async def terminal_screen(self, args: dict[str, Any]) -> dict[str, Any]:
        entry = self._find_terminal(int(args.get("session_id", 0) or 0), str(args.get("session_name", "")))
        if entry is None:
            return {"ok": False, "error": "interactive terminal session not found"}
        job_id, session_name, terminal = entry
        if args.get("rows") or args.get("cols"):
            terminal.resize(int(args.get("rows", terminal.rows)), int(args.get("cols", terminal.cols)))
        wait_seconds = float(args.get("wait_seconds", 0) or 0)
        if wait_seconds > 0:
            terminal.wait_for_change(min(wait_seconds, 30))
        terminal.drain()
        return {
            "ok": True,
            "terminal": True,
            "session_id": job_id,
            "session_name": session_name,
            "output": terminal.get_status(),
        }

    async def terminal_close(self, args: dict[str, Any]) -> dict[str, Any]:
        entry = self._find_terminal(int(args.get("session_id", 0) or 0), str(args.get("session_name", "")))
        if entry is None:
            return {"ok": False, "error": "interactive terminal session not found"}
        job_id, session_name, terminal = entry
        terminal.close()
        self.terminal_jobs.pop(job_id, None)
        return {"ok": True, "terminal": True, "session_id": job_id, "session_name": session_name, "output": "closed"}

    async def view_image(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_image_path(str(args["path"]))
        if not path.is_file():
            return {"ok": False, "error": "image file does not exist", "path": str(args["path"])}
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        info: dict[str, Any] = {
            "ok": True,
            "path": self._display_path(path),
            "mime": mime,
            "size": path.stat().st_size,
        }
        try:
            import base64
            import io

            from PIL import Image, ImageStat

            # Allow reasonably large images but keep decompression bomb protection
            Image.MAX_IMAGE_PIXELS = 200_000_000  # ~200 megapixels
            try:
                img = Image.open(path)
            except Image.DecompressionBombError:
                return {
                    "ok": False,
                    "error": "Image too large to process safely (decompression bomb protection). Use a smaller image or disable protection manually.",
                    "path": self._display_path(path),
                }
            info.update({
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
            })

            # --- base64 thumbnail ---
            max_thumb = 512
            thumb = img.copy()
            thumb.thumbnail((max_thumb, max_thumb), Image.LANCZOS)
            if thumb.mode in ("RGBA", "P"):
                rgb = Image.new("RGB", thumb.size, (255, 255, 255))
                if thumb.mode == "P":
                    thumb = thumb.convert("RGBA")
                rgb.paste(thumb, mask=thumb.split()[-1] if thumb.mode == "RGBA" else None)
                thumb = rgb
            elif thumb.mode not in ("RGB", "L"):
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=75)
            info["base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
            info["base64_mime"] = "image/jpeg"
            info["base64_note"] = f"{thumb.width}x{thumb.height} JPEG thumbnail"

            # --- content analysis ---
            analysis_parts: list[str] = []
            w, h = img.width, img.height
            if w > h * 1.5:
                analysis_parts.append("wide landscape orientation")
            elif h > w * 1.5:
                analysis_parts.append("tall portrait orientation")
            elif abs(w - h) < max(w, h) * 0.05:
                analysis_parts.append("square format")
            else:
                analysis_parts.append("standard orientation")

            if img.mode == "L":
                stat = ImageStat.Stat(thumb)
                mean_brightness = stat.mean[0]
                analysis_parts.append("grayscale")
                if mean_brightness < 85:
                    analysis_parts.append("very dark overall")
                elif mean_brightness > 170:
                    analysis_parts.append("very bright overall")
                else:
                    analysis_parts.append("medium brightness")
            else:
                rgb_small = thumb if thumb.mode == "RGB" else thumb.convert("RGB")
                stat = ImageStat.Stat(rgb_small)
                r, g, b = stat.mean[0], stat.mean[1], stat.mean[2]
                mean_brightness = (r + g + b) / 3
                if mean_brightness < 85:
                    analysis_parts.append("very dark overall")
                elif mean_brightness > 170:
                    analysis_parts.append("very bright overall")
                else:
                    analysis_parts.append("medium brightness")

                # Dominant colors
                quantized = rgb_small.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
                palette = quantized.getpalette()[:18]
                hist = quantized.histogram()
                total = sum(hist)
                color_summary = []
                for ci in range(6):
                    pct = hist[ci] / total * 100 if total > 0 else 0
                    if pct >= 8:
                        cr, cg, cb = palette[ci * 3], palette[ci * 3 + 1], palette[ci * 3 + 2]
                        color_summary.append(f"#{cr:02x}{cg:02x}{cb:02x} ~{pct:.0f}%")
                if color_summary:
                    analysis_parts.append("dominant colors: " + ", ".join(color_summary))

                std_r, std_g, std_b = stat.stddev[0], stat.stddev[1], stat.stddev[2]
                avg_std = (std_r + std_g + std_b) / 3
                if avg_std > 70:
                    analysis_parts.append("high color contrast (likely photo or detailed graphic)")
                elif avg_std < 25:
                    analysis_parts.append("low color variety (likely screenshot, diagram, or flat design)")
                else:
                    analysis_parts.append("moderate color variety")

            info["analysis"] = "; ".join(analysis_parts)
            img.close()

        except ImportError:
            info["note"] = "Install Pillow (pip install Pillow) for image analysis"
        except Exception as exc:
            info["analysis_error"] = f"{type(exc).__name__}: {exc}"
        return info

    async def web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        max_results = max(1, min(int(args.get("max_results", 10)), 20))
        region = str(args.get("region", "wt-wt")).strip() or "wt-wt"
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return {"ok": False, "error": "ddgs package is not installed. Run: pip install ddgs"}
        results: list[dict[str, str]] = []
        try:
            with DDGS() as ddgs:
                for entry in ddgs.text(query, max_results=max_results, region=region):
                    results.append({
                        "title": str(entry.get("title", "")),
                        "url": str(entry.get("href", "")),
                        "body": str(entry.get("body", "")),
                    })
        except Exception as exc:
            return {"ok": False, "error": f"Search failed: {type(exc).__name__}: {exc}"}
        return {"ok": True, "query": query, "results": results, "count": len(results)}

    def _extract_structured(self, html: str, url: str, max_chars: int) -> dict[str, Any]:
        from ._html_extract import extract_structured
        return extract_structured(html, url, max_chars)

    async def web_extract(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url", "")).strip()
        if not url:
            return {"ok": False, "error": "url is required"}
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        max_chars = max(500, min(int(args.get("max_chars", 5000)), 30000))
        try:
            from curl_cffi import requests
        except ImportError:
            return {"ok": False, "error": "Missing dependency: curl_cffi. Run: pip install curl_cffi readability-lxml html2text beautifulsoup4"}
        try:
            resp = requests.get(
                url,
                impersonate="chrome131",
                timeout=15,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
                },
            )
            if resp.status_code != 200:
                return {"ok": False, "error": f"HTTP {resp.status_code}", "url": url}
        except Exception as exc:
            return {"ok": False, "error": f"Fetch failed: {type(exc).__name__}: {exc}", "url": url}
        try:
            data = self._extract_structured(resp.text, url, max_chars)
        except Exception as exc:
            return {"ok": False, "error": f"Extraction failed: {type(exc).__name__}: {exc}", "url": url}
        return {
            "ok": True,
            "url": str(resp.url),
            **data,
        }

    async def apply_patch(self, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("path"):
            return self._apply_simple_edit(args)
        patch = str(args.get("patch", ""))
        if not patch.strip():
            return {
                "ok": False,
                "error": "Use path with old_text and new_text for a normal edit, or pass patch for an advanced edit.",
            }
        if "*** Begin Patch" in patch:
            return self._apply_codex_patch(patch)
        return self._apply_unified_patch(patch)

    def _apply_simple_edit(self, args: dict[str, Any]) -> dict[str, Any]:
        relative = str(args["path"])
        path = self._resolve(relative)
        if bool(args.get("delete")):
            if not path.is_file():
                return {"ok": False, "path": relative, "error": "Delete target does not exist"}
            before = path.read_text(encoding="utf-8", errors="replace")
            return self._commit_changes([(path, before, None)])

        new_text = str(args.get("new_text", ""))
        if "old_text" not in args:
            if path.exists():
                return {
                    "ok": False,
                    "path": relative,
                    "error": "File already exists. Pass old_text and new_text to replace exact text, or set delete=true to remove it.",
                }
            return self._commit_changes([(path, "", new_text)])

        if not path.is_file():
            return {"ok": False, "path": relative, "error": "Update target does not exist"}
        before = path.read_text(encoding="utf-8", errors="replace")
        old_text = str(args["old_text"])
        matches = before.count(old_text)
        if matches != 1:
            reason = "not found" if matches == 0 else f"ambiguous ({matches} matches)"
            return {
                "ok": False,
                "path": relative,
                "error": f"old_text {reason}. Read the exact surrounding text and retry with a unique old_text value.",
            }
        return self._commit_changes([(path, before, before.replace(old_text, new_text, 1))])

    def _apply_unified_patch(self, patch: str) -> dict[str, Any]:
        chunks = split_multi_file_diff(patch)
        if not chunks:
            return {"ok": False, "error": "Could not find target files in unified diff"}
        changes: list[tuple[Path, str, str | None]] = []
        for relative, file_patch in chunks.items():
            path = self._resolve(relative)
            before = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            after = apply_unified_patch_to_text(before, file_patch)
            changes.append((path, before, after))
        return self._commit_changes(changes)

    def _apply_codex_patch(self, patch: str) -> dict[str, Any]:
        lines = patch.splitlines()
        try:
            index = lines.index("*** Begin Patch") + 1
            end = lines.index("*** End Patch")
        except ValueError:
            return {"ok": False, "error": "Patch must contain *** Begin Patch and *** End Patch"}
        changes: list[tuple[Path, str, str | None]] = []
        # Track cumulative file state so multiple Update File sections
        # targeting the same file keep each other's changes.
        file_state: dict[Path, str] = {}
        while index < end:
            header = lines[index]
            if not header.startswith("*** "):
                return {"ok": False, "error": f"Unexpected patch line: {header}"}
            if header.startswith("*** Add File: "):
                relative = header.split(": ", 1)[1].strip()
                path = self._resolve(relative)
                if path.exists():
                    return {"ok": False, "path": relative, "error": "Add File target already exists"}
                body: list[str] = []
                index += 1
                while index < end and not lines[index].startswith("*** "):
                    if not lines[index].startswith("+"):
                        return {"ok": False, "path": relative, "error": "Add File lines must start with +"}
                    body.append(lines[index][1:])
                    index += 1
                after = "\n".join(body) + ("\n" if body else "")
                file_state[path] = after
                changes.append((path, "", after))
                continue
            if header.startswith("*** Delete File: "):
                relative = header.split(": ", 1)[1].strip()
                path = self._resolve(relative)
                if not path.is_file():
                    return {"ok": False, "path": relative, "error": "Delete File target does not exist"}
                before = file_state.pop(path, None)
                if before is None:
                    before = path.read_text(encoding="utf-8", errors="replace")
                changes.append((path, before, None))
                index += 1
                continue
            if header.startswith("*** Update File: "):
                relative = header.split(": ", 1)[1].strip()
                source = self._resolve(relative)
                before = file_state.get(source)
                if before is None:
                    if not source.is_file():
                        return {"ok": False, "path": relative, "error": "Update File target does not exist"}
                    before = source.read_text(encoding="utf-8", errors="replace")
                index += 1
                move_to: str | None = None
                if index < end and lines[index].startswith("*** Move to: "):
                    move_to = lines[index].split(": ", 1)[1].strip()
                    index += 1
                section: list[str] = []
                while index < end and not lines[index].startswith("*** "):
                    section.append(lines[index])
                    index += 1
                after = apply_codex_update_to_text(before, section)
                target = self._resolve(move_to) if move_to else source
                if target != source and target.exists():
                    return {"ok": False, "path": move_to, "error": "Move target already exists"}
                orig_before = file_state.get(target, "" if target != source else before)
                if target not in file_state and target != source and target.is_file():
                    # Target file existed before this patch — capture its original content.
                    orig_before = target.read_text(encoding="utf-8", errors="replace")
                file_state[target] = after
                if target != source:
                    file_state.pop(source, None)
                    changes.append((source, before, None))
                # Replace any previous change entry for this target so the diff
                # compares the original content with the final accumulated result.
                changes = [(p, b, a) for (p, b, a) in changes if p != target]
                changes.append((target, orig_before, after))
                continue
            return {"ok": False, "error": f"Unknown patch header: {header}"}
        return self._commit_changes(changes)

    def _commit_changes(self, changes: list[tuple[Path, str, str | None]]) -> dict[str, Any]:
        changed_paths: list[str] = []
        diffs: list[str] = []
        # Git safety net: auto-stash before modifying files
        if self.git_safety and self._is_git_repo():
            self._git_stash_safety()
        # Record undo snapshots before modifying anything
        undo_entry: list[tuple[Path, str | None]] = []
        for path, before, after in changes:
            if after is None:
                # Deleting file: save content to restore on undo
                if path.is_file():
                    undo_entry.append((path, before))
                else:
                    undo_entry.append((path, None))
            elif before != after:
                if path.is_file():
                    undo_entry.append((path, before))
                else:
                    # Creating new file
                    undo_entry.append((path, None))
            else:
                continue
        for path, before, after in changes:
            relative = self._relative(path)
            if after is None:
                path.unlink()
            elif before != after:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(after, encoding="utf-8")
            else:
                continue
            changed_paths.append(relative)
            diffs.append(make_unified_diff(relative, before, after or ""))
        if undo_entry:
            self._undo_stack.append(undo_entry)
        return {
            "ok": True,
            "paths": changed_paths,
            "path": changed_paths[0] if changed_paths else "",
            "changed": bool(changed_paths),
            "diff": "\n".join(item for item in diffs if item),
        }

    def undo_last_change(self) -> dict[str, Any]:
        """Undo the most recent patch operation. Returns info about what was reverted."""
        if not self._undo_stack:
            return {"ok": False, "error": "Nothing to undo", "paths": []}
        entry = self._undo_stack.pop()
        reverted: list[str] = []
        for path, previous_content in entry:
            relative = self._relative(path)
            if previous_content is None:
                # File was created by the patch — delete it
                if path.is_file():
                    path.unlink()
                reverted.append(f"{relative} (removed)")
            else:
                # File was modified or deleted — restore previous content
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(previous_content, encoding="utf-8")
                reverted.append(relative)
        return {
            "ok": True,
            "paths": reverted,
            "undo_depth_remaining": len(self._undo_stack),
        }
