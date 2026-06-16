from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time
import tty
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pyte


@dataclass
class CommandResult:
    command: str
    cwd: str
    returncode: int
    output: str


KEY_SEQUENCES = {
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "space": " ",
    "backspace": "\x7f",
    "delete": "\x1b[3~",
    "insert": "\x1b[2~",
    "escape": "\x1b",
    "esc": "\x1b",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "pageup": "\x1b[5~",
    "pagedown": "\x1b[6~",
    "ctrl+up": "\x1b[1;5A",
    "ctrl+down": "\x1b[1;5B",
    "ctrl+right": "\x1b[1;5C",
    "ctrl+left": "\x1b[1;5D",
    "shift+up": "\x1b[1;2A",
    "shift+down": "\x1b[1;2B",
    "shift+right": "\x1b[1;2C",
    "shift+left": "\x1b[1;2D",
    "alt+up": "\x1b[1;3A",
    "alt+down": "\x1b[1;3B",
    "alt+right": "\x1b[1;3C",
    "alt+left": "\x1b[1;3D",
    "f1": "\x1bOP",
    "f2": "\x1bOQ",
    "f3": "\x1bOR",
    "f4": "\x1bOS",
    "f5": "\x1b[15~",
    "f6": "\x1b[17~",
    "f7": "\x1b[18~",
    "f8": "\x1b[19~",
    "f9": "\x1b[20~",
    "f10": "\x1b[21~",
    "f11": "\x1b[23~",
    "f12": "\x1b[24~",
}
for _index, _char in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
    KEY_SEQUENCES[f"ctrl+{_char}"] = chr(_index)
for _char in "abcdefghijklmnopqrstuvwxyz":
    KEY_SEQUENCES[f"alt+{_char}"] = f"\x1b{_char}"
KEY_SEQUENCES["ctrl+_"] = "\x1f"
KEY_SEQUENCES["ctrl+/"] = "\x1f"


class InteractiveTerminal:
    """Background PTY with a model-readable virtual screen."""

    def __init__(self, command: str, cwd: Path, rows: int = 40, cols: int = 140) -> None:
        self.command = command
        self.cwd = cwd
        self.rows = rows
        self.cols = cols
        self.pid: int | None = None
        self.fd: int | None = None
        self.alive = False
        self.returncode: int | None = None
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.previous_screen = ""

    def start(self) -> None:
        shell = os.environ.get("SHELL", "/bin/bash")
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(self.cwd)
            env = os.environ.copy()
            env.update({"TERM": "xterm-256color", "COLUMNS": str(self.cols), "LINES": str(self.rows)})
            os.execvpe(shell, [shell, "-lc", self.command], env)
            raise SystemExit(127)
        self.pid = pid
        self.fd = fd
        self.alive = True
        self.resize(self.rows, self.cols)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        time.sleep(0.1)
        self.drain()

    def drain(self, timeout: float = 0.25) -> None:
        if self.fd is None:
            return
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.fd], [], [], 0.03)
            if not readable:
                continue
            try:
                chunk = os.read(self.fd, 65536)
            except OSError:
                self.is_alive()
                break
            if not chunk:
                self.alive = False
                break
            self.stream.feed(chunk.decode("utf-8", errors="replace"))

    def resize(self, rows: int, cols: int) -> None:
        self.rows = max(5, min(int(rows), 200))
        self.cols = max(20, min(int(cols), 400))
        self.screen.resize(self.rows, self.cols)
        if self.fd is not None:
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))

    def send(self, text: str, wait: float = 0.15) -> None:
        if self.fd is None or not self.is_alive():
            raise ValueError("terminal session has exited")
        os.write(self.fd, encode_terminal_input(text))
        time.sleep(max(0.0, wait))
        self.drain()

    def wait_for_change(self, timeout: float = 5.0) -> bool:
        before = self.get_screen()
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            self.drain(0.2)
            if self.get_screen() != before:
                return True
        return False

    def get_screen(self) -> str:
        lines = [line.rstrip() for line in self.screen.display]
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines) if lines else "(empty screen)"

    def get_status(self) -> str:
        current = self.get_screen()
        changed = current != self.previous_screen
        self.previous_screen = current
        cursor = self.screen.cursor
        numbered = "\n".join(
            f"{index:>{len(str(len(current.splitlines())))}}| {line}"
            for index, line in enumerate(current.splitlines(), start=1)
        )
        return (
            f"APP: {self.detect_app()}\n"
            f"ALIVE: {str(self.is_alive()).lower()}\n"
            f"RETURN_CODE: {self.returncode if self.returncode is not None else 'running'}\n"
            f"CURSOR: line {cursor.y + 1}, col {cursor.x + 1}\n"
            f"CHANGED: {str(changed).lower()}\n"
            f"SCREEN:\n{numbered or '(empty screen)'}"
        )

    def detect_app(self) -> str:
        screen = self.get_screen().lower()
        if "gnu nano" in screen:
            return "nano"
        if any(marker in screen for marker in ("-- insert --", "-- visual --", "-- replace --")):
            return "vim"
        if ">>>" in screen:
            return "python-repl"
        if "htop" in screen[:300]:
            return "htop"
        if "top -" in screen[:300]:
            return "top"
        return "terminal"

    def is_alive(self) -> bool:
        if not self.alive or self.pid is None:
            return False
        try:
            finished, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self.alive = False
            return False
        if finished:
            self.alive = False
            self.returncode = os.waitstatus_to_exitcode(status)
        return self.alive

    def close(self) -> None:
        if self.pid is not None and self.is_alive():
            with suppress(ProcessLookupError):
                os.killpg(self.pid, signal.SIGTERM)
            time.sleep(0.05)
            if self.is_alive():
                with suppress(ProcessLookupError):
                    os.killpg(self.pid, signal.SIGKILL)
        if self.fd is not None:
            with suppress(OSError):
                os.close(self.fd)
        self.fd = None
        self.pid = None
        self.alive = False


def encode_terminal_input(text: str) -> bytes:
    chunks: list[str] = []
    offset = 0
    for match in re.finditer(r"<([^>]+)>", text):
        chunks.append(text[offset:match.start()])
        key = match.group(1).strip().lower()
        sequence = _terminal_sequence(key)
        chunks.append(sequence if sequence is not None else match.group(0))
        offset = match.end()
    chunks.append(text[offset:])
    return "".join(chunks).encode("utf-8")


def _terminal_sequence(key: str) -> str | None:
    if key.startswith("click "):
        try:
            _click, row, col = key.split()
            return f"\x1b[<0;{int(col)};{int(row)}M\x1b[<0;{int(col)};{int(row)}m"
        except (ValueError, TypeError):
            return None
    repeat = 1
    if "*" in key:
        key, raw_repeat = key.rsplit("*", 1)
        try:
            repeat = max(1, min(int(raw_repeat), 200))
        except ValueError:
            return None
    sequence = KEY_SEQUENCES.get(key)
    return sequence * repeat if sequence is not None else None


async def capture_command(command: str, cwd: Path, timeout: int = 120) -> CommandResult:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            executable=os.environ.get("SHELL", "/bin/bash"),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace") if stdout else ""
            return CommandResult(
                command=command,
                cwd=str(cwd),
                returncode=proc.returncode or 0,
                output=output,
            )
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
                output = stdout.decode(errors="replace") if stdout else ""
            except Exception:
                output = ""
            return CommandResult(command=command, cwd=str(cwd), returncode=124, output=output)
    except Exception as exc:
        return CommandResult(command=command, cwd=str(cwd), returncode=1, output=f"{type(exc).__name__}: {exc}")


def run_foreground_pty(command: str, cwd: Path) -> int:
    """Run a command with a real PTY, handing the user's terminal to it.

    This is for complex TUIs such as vim, htop, curses apps, or custom shells.
    """
    import shutil as _shutil
    term_size = _shutil.get_terminal_size()
    shell = os.environ.get("SHELL", "/bin/bash")
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        env = os.environ.copy()
        env.update({"COLUMNS": str(term_size.columns), "LINES": str(term_size.lines)})
        os.execvpe(shell, [shell, "-lc", command], env)
        raise SystemExit(127)

    old_settings = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            readable, _, _ = select.select([sys.stdin.fileno(), fd], [], [])
            if sys.stdin.fileno() in readable:
                data = os.read(sys.stdin.fileno(), 4096)
                if not data:
                    break
                os.write(fd, data)
            if fd in readable:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        with suppress(OSError):
            os.close(fd)

    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1
