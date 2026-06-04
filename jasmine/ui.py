from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import textwrap
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import create_output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import (
    BlockQuote,
    CodeBlock,
    Heading,
    HorizontalRule,
    ListElement,
    ListItem,
    Markdown,
    Paragraph,
    TableBodyElement,
    TableDataElement,
    TableElement,
    TableHeaderElement,
    TableRowElement,
)
from rich.spinner import Spinner
from rich._spinners import SPINNERS

# Inject custom star spinner into Rich's registry
SPINNERS["star"] = {"frames": ["·", "•", "✦", "★"], "interval": 140}
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme as RichTheme


COMMAND_ROWS: list[tuple[str, str]] = [
    ("/menu", "open command menu"),
    ("/provider", "switch provider"),
    ("/resume", "resume a past conversation"),
    ("/tools", "show available tools"),
    ("/trusted", "show trusted shell prefixes"),
    ("/usage", "show API token usage"),
    ("/exit", "quit Jasmine"),
    ("/clear", "clear chat context"),
]

SUCCESS_STYLE = "bold #22c55e"
DIM_STYLE = "#6b6b6b italic"
USER_MESSAGE_PREVIEW_LIMIT = 720
MODIFIED_KEYBOARD_ENABLE = "\x1b[>1u"
MODIFIED_KEYBOARD_DISABLE = "\x1b[<u"
MARKDOWN_THEME = RichTheme(
    {
        "markdown.block_quote": "italic #b8b8b8",
        "markdown.code": "bold #e5e7eb on #252525",
        "markdown.code_block": "#d1d5db on #202020",
        "markdown.h1": "bold underline #f5f5f5",
        "markdown.h2": "bold underline #f5f5f5",
        "markdown.h3": "bold underline #e5e7eb",
        "markdown.h4": "bold underline #d1d5db",
        "markdown.h5": "bold underline #d1d5db",
        "markdown.h6": "bold underline #b8b8b8",
        "markdown.hr": "#737373",
        "markdown.item.bullet": SUCCESS_STYLE,
        "markdown.item.number": SUCCESS_STYLE,
        "markdown.link": "bold underline #e5e7eb",
        "markdown.link_url": "bold underline #e5e7eb",
        "markdown.list": "#d1d5db",
        "markdown.table.border": "#737373",
        "markdown.table.header": "bold underline #e5e7eb",
    }
)


class _CompactParagraph(Paragraph):
    new_line = True


class _CompactHeading(Heading):
    new_line = True


class _CompactHorizontalRule(HorizontalRule):
    def __rich_console__(self, console: Console, options):  # type: ignore[no-untyped-def]
        yield Text("─" * max(1, options.max_width), style=console.get_style("markdown.hr", default="none"))


class _CompactCodeBlock(CodeBlock):
    new_line = True

    def __rich_console__(
        self, console: Console, options: Any
    ) -> Any:
        from rich.syntax import Syntax

        code = str(self.text).rstrip()
        syntax = Syntax(
            code, self.lexer_name, theme=self.theme, word_wrap=True, padding=0
        )
        yield syntax


class _CompactBlockQuote(BlockQuote):
    new_line = False


class _CompactListElement(ListElement):
    new_line = False


class _CompactListItem(ListItem):
    new_line = False


class _CompactTableElement(TableElement):
    new_line = False


class _CompactTableBodyElement(TableBodyElement):
    new_line = False


class _CompactTableHeaderElement(TableHeaderElement):
    new_line = False


class _CompactTableRowElement(TableRowElement):
    new_line = False


class _CompactTableDataElement(TableDataElement):
    new_line = False


class CompactMarkdown(Markdown):
    elements = {
        **Markdown.elements,
        "paragraph_open": _CompactParagraph,
        "heading_open": _CompactHeading,
        "hr": _CompactHorizontalRule,
        "fence": _CompactCodeBlock,
        "code_block": _CompactCodeBlock,
        "blockquote_open": _CompactBlockQuote,
        "bullet_list_open": _CompactListElement,
        "ordered_list_open": _CompactListElement,
        "list_item_open": _CompactListItem,
        "table_open": _CompactTableElement,
        "tbody_open": _CompactTableBodyElement,
        "thead_open": _CompactTableHeaderElement,
        "tr_open": _CompactTableRowElement,
        "td_open": _CompactTableDataElement,
        "th_open": _CompactTableDataElement,
    }


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        if " " in text:
            return
        rows = COMMAND_ROWS if text == "/" else [row for row in COMMAND_ROWS if row[0].startswith(text)]
        for command, desc in rows:
            yield Completion(command, start_position=-len(text), display=command, display_meta=desc)


class ApprovalCompleter(Completer):
    rows = [
        ("allow", "run this command once"),
        ("deny", "stop this turn"),
        ("remember", "trust a safe prefix next time"),
    ]

    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor.strip().lower()
        for command, desc in self.rows:
            if command.startswith(text):
                yield Completion(command, start_position=-len(document.text_before_cursor), display=command, display_meta=desc)


@dataclass
class PlanItem:
    text: str
    status: str = "pending"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string like '1min 21seconds' or '34seconds'."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    parts: list[str] = []
    if minutes:
        parts.append(f"{minutes}min")
    if secs or not parts:
        parts.append(f"{secs}second" if secs == 1 else f"{secs}seconds")
    return " ".join(parts)


class AssistantStream:
    def __init__(self, ui: "TerminalUI") -> None:
        self.ui = ui
        self.buffer = ""
        self.thinking_buffer = ""
        self.followed_by_tools = False
        self._live: Live | None = None
        self._start_time = time.time()
        self._thinking_start: float | None = None

    async def __aenter__(self) -> "AssistantStream":
        self.ui.open_block("assistant")
        if self.ui.console.is_terminal:
            self._live = Live(self._render(), console=self.ui.console, refresh_per_second=16, transient=False, vertical_overflow="crop")
            self._live.start()
            self.ui._register_live(self._live)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._live:
            with suppress(Exception):
                self._live.update(self._render(final=True))
                self._live.stop()
            self.ui._unregister_live(self._live)
        elif self.buffer.strip() or self.thinking_buffer.strip():
            self.ui.console.print(self._render(final=True))
        if self.buffer.strip():
            self.ui._printed_something = True
            if not self.followed_by_tools:
                self.ui.console.print()
        else:
            self.ui._last_block = "tool"

    def append(self, text: str) -> None:
        if not text:
            return
        self.buffer += text
        if self._live:
            self._live.update(self._render())

    def append_thinking(self, text: str) -> None:
        if not text:
            return
        if self._thinking_start is None and not self.thinking_buffer:
            self._thinking_start = time.time()
        self.thinking_buffer += text
        if self._live:
            self._live.update(self._render())

    def set_followed_by_tools(self, followed_by_tools: bool) -> None:
        self.followed_by_tools = followed_by_tools

    def _render(self, final: bool = False) -> RenderableType:
        parts: list[RenderableType] = []
        has_text = bool(self.buffer.strip())
        has_thinking = bool(self.thinking_buffer.strip())

        if not final:
            label = "generating" if has_text else "thinking"
            parts.append(Spinner("star", Text(label, style="bold #a3a3a3"), style=SUCCESS_STYLE, speed=1.0))

        if has_thinking:
            if final and has_text:
                elapsed = time.time() - (self._thinking_start or self._start_time)
                parts.append(Text(f"thought for {_format_duration(elapsed)}", style=DIM_STYLE))
                parts.append(Text(""))
            elif not final and not has_text:
                think = self.thinking_buffer.strip()
                lines = think.splitlines()
                if len(lines) > 2:
                    lines = lines[-2:]
                preview = "\n".join(lines)
                parts.append(Text(preview, style="#7a7a7a italic"))

        if has_text:
            parts.append(CompactMarkdown(self.buffer))

        if parts:
            return Group(*parts) if len(parts) > 1 else parts[0]
        return Text("")


class ToolActivity:
    def __init__(self, ui: "TerminalUI", name: str, args: dict[str, Any]) -> None:
        self.ui = ui
        self.name = name
        self.args = args
        self._live: Live | None = None

    async def __aenter__(self) -> "ToolActivity":
        self.ui.open_block("tool")
        if self.ui.console.is_terminal:
            self._live = Live(self._render(), console=self.ui.console, refresh_per_second=30, transient=True, vertical_overflow="crop")
            self._live.start()
            self.ui._register_live(self._live)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._live:
            with suppress(Exception):
                self._live.stop()
            self.ui._unregister_live(self._live)

    def _render(self) -> RenderableType:
        return Spinner("star", self.ui._tool_activity_renderable(self.name, self.args), style=SUCCESS_STYLE, speed=1.0)



def _effective_width() -> int:
    """Usable columns per visual row, accounting for the ``▌ `` prompt."""
    app = get_app()
    return max(1, app.output.get_size().columns - 2)


def visual_up(buffer: Any) -> None:
    """Move up one visual row, handling soft-wrapped lines.

    Within a wrapped line, step up one visual segment.  On the first
    visual row of a line, jump to the previous logical line.  From the
    very first position, fall back to history.
    """
    doc = buffer.document
    row = doc.cursor_position_row
    col = doc.cursor_position_col
    width = _effective_width()

    # Still on a wrapped segment of the current line?
    if col >= width:
        buffer.cursor_position = doc.translate_row_col_to_index(row, col - width)
        return

    # First visual row — go to the previous logical line.
    if row > 0:
        buffer.cursor_up()
        return

    # Top of the buffer — history.
    buffer.history_backward()


def visual_down(buffer: Any) -> None:
    """Move down one visual row, handling soft-wrapped lines.

    On the *first* visual row of any line, prefer the next logical line
    so that pressing Enter to create a new line never traps the cursor
    inside a long wrapped line.  Only when no next logical line exists
    do we step down within the current (last) line's wraps.  From the
    very last position, fall forward in history.
    """
    doc = buffer.document
    row = doc.cursor_position_row
    col = doc.cursor_position_col
    width = _effective_width()
    last_row = len(doc.lines) - 1

    if row < len(doc.lines):
        line_len = len(doc.lines[row])
    else:
        line_len = 0

    # First visual row AND a next logical line exists — jump to it.
    if col < width and row < last_row:
        buffer.cursor_down()
        return

    # Not on the last visual row — step down within the current line.
    if col + width < line_len:
        buffer.cursor_position = doc.translate_row_col_to_index(row, col + width)
        return

    # Last visual row but more logical lines — next line.
    if row < last_row:
        buffer.cursor_down()
        return

    # Bottom of the buffer — history.
    buffer.history_forward()


class _UnifiedCompleter(Completer):
    """Slash commands without interfering with chat text."""
    def __init__(self) -> None:
        self._slash = SlashCommandCompleter()

    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor
        if text.startswith("/"):
            yield from self._slash.get_completions(document, complete_event)

class TerminalUI:
    """Rich terminal presentation designed to stay stable with prompt_toolkit.

    The UI intentionally avoids full-screen alternate-buffer tricks. Everything is
    durable scrollback: user messages, tool cards, shell output previews, and diffs
    keep their background blocks when the user scrolls back.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.console = Console(highlight=False, theme=MARKDOWN_THEME)
        self.root = root.resolve() if root else Path.cwd().resolve()
        self.plan_items: list[PlanItem] = []
        self._last_plan_signature = ""
        self._last_block: str | None = None
        self._printed_something = False
        self._theme = None
        self._slash_completion_navigated = False
        self._active_lives: list[Live] = []
        self._aux_prompt_depth = 0
        self._redraw_callback: Callable[[], None] | None = None
        self._interrupt_handler: Any = None
        self.trusted_prefixes: list[str] = []
        self.trusted_path = self.root / ".jasmine" / "trusted_commands.json"
        self._load_trusted_prefixes()

        os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")
        self._pt_output = create_output()
        if hasattr(self._pt_output, "enable_cpr"):
            self._pt_output.enable_cpr = False

        self._history = InMemoryHistory()

        self.session = PromptSession(
            multiline=True,
            enable_history_search=True,
            mouse_support=False,
            key_bindings=self._prompt_bindings(),
            completer=_UnifiedCompleter(),
            complete_while_typing=True,
            complete_style=CompleteStyle.MULTI_COLUMN,
            reserve_space_for_menu=4,
            output=self._pt_output,
            style=self._style(),
            history=self._history,
            erase_when_done=True,
            prompt_continuation=self._prompt_continuation,
        )
        self.approval_session = PromptSession(
            multiline=False,
            mouse_support=False,
            key_bindings=self._approval_bindings(),
            completer=ApprovalCompleter(),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=6,
            output=self._pt_output,
            style=self._style(),
        )
        self.line_session = PromptSession(multiline=False, mouse_support=False, output=self._pt_output, style=self._style())

    def _prompt_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        has_slash_completions = Condition(
            lambda: bool(
                get_app().current_buffer.text.startswith("/")
                and get_app().current_buffer.complete_state
            )
        )

        @kb.add("/")
        def _(event):  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            buffer.insert_text("/")
            self._slash_completion_navigated = False
            if buffer.document.text_before_cursor == "/":
                self._refresh_slash_completion(buffer)

        @kb.add("enter")
        def _(event):  # type: ignore[no-untyped-def]
            """Enter: newline for chat, send for slash commands."""
            buffer = event.current_buffer
            text = buffer.text
            if text.startswith("/") and "\n" not in text:
                # Apply completion if menu is visible
                state = buffer.complete_state
                if state and state.completions:
                    comp = state.current_completion or state.completions[0]
                    buffer.apply_completion(comp)
                buffer.validate_and_handle()
            else:
                buffer.insert_text("\n")

        @kb.add("escape", "enter")
        def _(event):  # type: ignore[no-untyped-def]
            """Esc+Enter sends the message."""
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "c-m")
        def _(event):  # type: ignore[no-untyped-def]
            """Esc+Enter (carriage-return variant) sends the message."""
            event.current_buffer.validate_and_handle()

        @kb.add("down", filter=has_slash_completions)
        def _(event):  # type: ignore[no-untyped-def]
            self._slash_completion_navigated = event.current_buffer.text.startswith("/")
            event.current_buffer.complete_next()

        @kb.add("up", filter=has_slash_completions)
        def _(event):  # type: ignore[no-untyped-def]
            self._slash_completion_navigated = event.current_buffer.text.startswith("/")
            event.current_buffer.complete_previous()

        @kb.add("down", filter=~has_slash_completions, eager=True)
        def _(event):  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            if buffer.complete_state:
                buffer.cancel_completion()
            visual_down(buffer)

        @kb.add("up", filter=~has_slash_completions, eager=True)
        def _(event):  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            if buffer.complete_state:
                buffer.cancel_completion()
            visual_up(buffer)

        @kb.add("backspace")
        def _(event):  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            if buffer.selection_state is not None:
                buffer.cut_selection()
            else:
                buffer.delete_before_cursor(count=1)
            self._slash_completion_navigated = False
            self._refresh_slash_completion(buffer)

        # Ctrl+C stops generation (handled via SIGINT), does nothing at prompt.
        # Ctrl+D clears text if present, quits if input is empty.
        @kb.add("c-c")
        def _(event):  # type: ignore[no-untyped-def]
            pass

        @kb.add("c-d")
        def _(event):  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            if buffer.text:
                buffer.text = ""
            else:
                event.app.exit(exception=EOFError())

        return kb

    def _prompt_continuation(self, _width: int, _line_number: int, _is_soft_wrap: bool) -> AnyFormattedText:
        return [("class:prompt.arrow", "▌ ")]

    def _refresh_slash_completion(self, buffer: Any) -> None:
        text = buffer.document.text_before_cursor
        if text.startswith("/") and "\n" not in text:
            buffer.start_completion(select_first=False)

    def _approval_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            state = buffer.complete_state
            if state and state.completions:
                buffer.apply_completion(state.current_completion or state.completions[0])
            buffer.validate_and_handle()

        @kb.add("down", filter=has_completions)
        def _(event):  # type: ignore[no-untyped-def]
            event.current_buffer.complete_next()

        @kb.add("up", filter=has_completions)
        def _(event):  # type: ignore[no-untyped-def]
            event.current_buffer.complete_previous()

        @kb.add("c-c")
        def _(event):  # type: ignore[no-untyped-def]
            event.app.exit(result="deny")

        @kb.add("c-d")
        def _(event):  # type: ignore[no-untyped-def]
            event.app.exit(exception=EOFError())

        return kb

    def _style(self) -> Style:
        return Style.from_dict(
            {
                "": "#d6d6d6",
                "text-area": "",
                "prompt": "#e5e7eb bold",
                "prompt.name": "#e5e7eb bold",
                "prompt.path": "#86efac bold",
                "prompt.arrow": "#d9d9d9",
                "user.label": "bold #111827 bg:#d9d9d9",
                "completion-menu": "bg:#252525 #d1d5db",
                "completion-menu.completion.current": "bg:#22c55e #ffffff bold",
                "completion-menu.meta": "#9ca3af",
                "completion-menu.meta.completion.current": "bg:#22c55e #eff6ff",
                "scrollbar.background": "bg:#171717",
                "scrollbar.button": "bg:#525252",
                "approval": "#f3f4f6 bold bg:#2a2a2a",
            }
        )

    def set_theme(self, theme) -> None:
        self._theme = theme
        style_dict = {
            "": f"{theme.assistant_text}",
            "text-area": "",
            "prompt": theme.prompt_name,
            "prompt.name": theme.prompt_name,
            "prompt.path": theme.prompt_path,
            "prompt.arrow": f"{theme.prompt_arrow}",
            "user.label": "bold #111827 bg:#d9d9d9",
            "completion-menu": theme.completion_bg,
            "completion-menu.completion.current": theme.completion_current,
            "completion-menu.meta": theme.completion_meta,
            "completion-menu.meta.completion.current": theme.completion_current,
            "scrollbar.background": theme.scrollbar_bg,
            "scrollbar.button": theme.scrollbar_button,
            "approval": "#f3f4f6 bold bg:#2a2a2a",
        }
        new_style = Style.from_dict(style_dict)
        self.session.style = new_style
        self.approval_session.style = new_style
        self.line_session.style = new_style

    def set_root(self, root: Path) -> None:
        self.root = root.resolve()
        self.trusted_path = self.root / ".jasmine" / "trusted_commands.json"
        self._load_trusted_prefixes()

    def clear_input_history(self) -> None:
        self._history = InMemoryHistory()
        self.session.history = self._history
        # The default_buffer is created once in __init__ and caches the old
        # history reference. Update it so up/down arrows see the new history.
        if hasattr(self.session, "default_buffer"):
            self.session.default_buffer.history = self._history

    def load_history_from_messages(self, messages: list[dict[str, Any]]) -> None:
        self.clear_input_history()
        seen: set[str] = set()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", ""))
            normalized = content.strip()
            if not normalized or normalized.startswith("/") or normalized.startswith("[pasted-content:"):
                continue
            if content not in seen:
                seen.add(content)
                self._history.append_string(content)

    def _load_trusted_prefixes(self) -> None:
        try:
            data = json.loads(self.trusted_path.read_text(encoding="utf-8"))
            prefixes = data.get("prefixes", []) if isinstance(data, dict) else []
            self.trusted_prefixes = [str(item) for item in prefixes if str(item).strip()]
        except Exception:
            self.trusted_prefixes = []

    def _save_trusted_prefixes(self) -> None:
        try:
            self.trusted_path.parent.mkdir(parents=True, exist_ok=True)
            self.trusted_path.write_text(json.dumps({"prefixes": self.trusted_prefixes}, indent=2), encoding="utf-8")
        except Exception:
            pass

    def banner(self, root: str) -> None:
        self._last_block = None
        width = max(40, self.console.width)
        title = Text(" jasmine ", style="bold #0a0a0a on #86efac")
        title.append("  " + root, style="bold #0a0a0a on #86efac")
        # Pad to full terminal width so the green bar extends edge to edge
        if title.cell_len < width:
            title.append(" " * (width - title.cell_len), style="on #86efac")
        self.console.print(title)
        self.console.print(Text(" Esc+Enter sends · Enter newline · / menu · Ctrl+D exits", style="#737373"))
        self.console.print()
        self._printed_something = True

    def clear_terminal(self) -> None:
        """Erase the viewport and scrollback before rebuilding the durable UI."""
        sequence = "\x1b[2J\x1b[3J\x1b[H"
        try:
            self._pt_output.write_raw(sequence)
            self._pt_output.flush()
        except Exception:
            sys.stdout.write(sequence)
            sys.stdout.flush()
        self._last_block = None
        self._printed_something = False

    def set_redraw_callback(self, callback: Callable[[], None]) -> None:
        self._redraw_callback = callback

    def handle_resize(self) -> None:
        """Refresh active widgets or rebuild the transcript after SIGWINCH."""
        for live in list(self._active_lives):
            with suppress(Exception):
                live.refresh()
        self._invalidate_prompt()
        if self._active_lives or self._aux_prompt_depth:
            return
        if self._redraw_callback:
            self._redraw_callback()
            self._invalidate_prompt()

    def redraw_transcript(self, root: str, messages: list[dict[str, Any]]) -> None:
        self.clear_terminal()
        self.banner(root)
        self.replay_messages(messages)

    def replay_messages(self, messages: list[dict[str, Any]]) -> None:
        """Render saved model messages using the same compact UI as live output."""
        self.plan_items = []
        self._last_plan_signature = ""
        tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        rendered = False
        for message in messages:
            role = str(message.get("role", ""))
            if role == "user":
                self.print_user_message(str(message.get("content", "")))
                rendered = True
                continue
            if role == "assistant":
                content = str(message.get("content", ""))
                if content.strip():
                    self.print_markdown(content)
                    rendered = True
                for tool_call in message.get("tool_calls", []) or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    tool_id = str(tool_call.get("id", ""))
                    if not tool_id:
                        continue
                    tool_calls[tool_id] = (
                        str(function.get("name", "")),
                        self._json_object(function.get("arguments")),
                    )
                continue
            if role != "tool":
                continue
            name, _args = tool_calls.get(str(message.get("tool_call_id", "")), ("tool", {}))
            result = self._json_object(message.get("content"))
            if name == "update_plan" and result.get("ok") is not False:
                self.update_plan(result.get("plan", []) or [])
            else:
                self.print_tool_result(name, result)
            rendered = True
        if rendered:
            self.console.print()
        self.reset_phase()

    def _json_object(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            value = json.loads(str(raw))
        except Exception:
            return {"ok": False, "error": str(raw)}
        return value if isinstance(value, dict) else {"value": value}

    def _register_live(self, live: Live) -> None:
        if live not in self._active_lives:
            self._active_lives.append(live)

    def _unregister_live(self, live: Live) -> None:
        with suppress(ValueError):
            self._active_lives.remove(live)

    def _invalidate_prompt(self) -> None:
        for session in (self.session, self.approval_session, self.line_session):
            app = getattr(session, "app", None)
            if app is not None and getattr(app, "is_running", False):
                with suppress(Exception):
                    app.invalidate()

    def open_block(self, kind: str, *, separated: bool = True) -> None:
        if separated and self._printed_something and kind != self._last_block:
            self.console.print()
        self._last_block = kind
        self._printed_something = True
    def reset_phase(self) -> None:
        self._last_block = None

    async def prompt(self) -> str:
        cwd_label = self.root.name or str(self.root)
        if not sys.stdin.isatty():
            return await asyncio.to_thread(input, f"jasmine {cwd_label} › ")
        fragments: AnyFormattedText = [
            ("class:user.label", " user "),
            ("", "\n"),
            ("class:prompt.arrow", "▌ "),
        ]
        self._set_modified_keyboard(enabled=True)
        try:
            with patch_stdout(raw=True):
                text = await self.session.prompt_async(fragments)
        finally:
            self._set_modified_keyboard(enabled=False)
            self._reinstall_interrupt_handler()
        self._last_block = "user"
        self._printed_something = True
        return text

    def _set_modified_keyboard(self, *, enabled: bool) -> None:
        """Ask compatible terminals to distinguish modified Enter keys."""
        try:
            self._pt_output.write_raw(MODIFIED_KEYBOARD_ENABLE if enabled else MODIFIED_KEYBOARD_DISABLE)
            self._pt_output.flush()
        except Exception:
            pass

    def print_user_message(self, content: str, max_lines: int | None = None) -> None:
        self.open_block("user")
        preview = self._user_message_preview(content, max_lines=max_lines)
        self._print_labeled_block("user", preview, fg="#f3f4f6", bg="#3b3b3b", label_style="bold #111827 on #d9d9d9", accent="#d9d9d9", wrap=True, max_lines=max_lines)

    def _user_message_preview(self, content: str, max_lines: int | None = None) -> str:
        # If max_lines is set, truncate to that many lines (with middle truncation)
        if max_lines is not None and max_lines > 0:
            raw_lines = content.splitlines()
            if len(raw_lines) > max_lines:
                head_count = max(1, max_lines // 2)
                tail_count = max(1, max_lines - head_count - 1)
                omitted = len(raw_lines) - head_count - tail_count
                head = "\n".join(raw_lines[:head_count])
                tail = "\n".join(raw_lines[-tail_count:])
                return f"{head}\n[truncated: {omitted} lines in middle]\n{tail}"
        # Original char-based truncation for backward compat
        if len(content) <= USER_MESSAGE_PREVIEW_LIMIT:
            return content
        excerpt_size = USER_MESSAGE_PREVIEW_LIMIT // 2
        head = content[:excerpt_size].rstrip()
        tail = content[-excerpt_size:].lstrip()
        return f"{head}\n[…]\n{tail}"

    def user_paste_notice(self, original_length: int, saved_path: str | None = None) -> None:
        suffix = f" · saved {saved_path}" if saved_path else ""
        self.console.print(f"[{SUCCESS_STYLE}]●[/] [bold #949494]pasted-content: {original_length} characters{suffix}[/]")

    def print_conversation_list(self, conversations: list[dict[str, Any]]) -> None:
        """Render the /resume conversation picker."""
        self._print_menu_rows(
            "Conversations",
            [
                (
                    str(idx),
                    f"{conv.get('mtime_label', '')} · {conv.get('message_count', 0)} msgs · {conv.get('subject', 'Untitled')}",
                )
                for idx, conv in enumerate(conversations, start=1)
            ],
        )

    def print_provider_list(self, providers: list[dict[str, str]]) -> None:
        self._print_menu_rows(
            "Providers",
            [
                (
                    str(idx),
                    f"{provider['name']} · {provider['model']} · {provider.get('vision', 'unknown')} · {provider['base_url']}",
                )
                for idx, provider in enumerate(providers, start=1)
            ],
        )

    async def line_prompt(self, label: str, style_class: str = "approval") -> str:
        """Simple single-line prompt for short text input (e.g. resume choice)."""
        if not sys.stdin.isatty():
            return await asyncio.to_thread(input, label)
        self._aux_prompt_depth += 1
        try:
            with patch_stdout(raw=True):
                return await self.line_session.prompt_async(
                    [("class:approval" if style_class == "approval" else "", label)]
                )
        finally:
            self._aux_prompt_depth -= 1
            self._reinstall_interrupt_handler()

    def _reinstall_interrupt_handler(self) -> None:
        """Re-install the asyncio SIGINT handler after a prompt_toolkit interaction.

        prompt_toolkit's Application replaces the asyncio-level SIGINT handler
        and then removes it on exit, leaving the self-pipe mechanism intact but
        with no Python callback.  We restore the callback here so that Ctrl+C
        during the next tool or stream can cancel the agent task.
        """
        if self._interrupt_handler is None:
            return
        import asyncio
        import signal as _signal
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(_signal.SIGINT, self._interrupt_handler)
        except (NotImplementedError, RuntimeError):
            pass

    def interrupt_notice(self) -> None:
        self.console.print("[yellow]●[/] [#9e9e9e]run interrupted[/#9e9e9e]")
        self.console.print()
    def print_markdown(self, content: str, *, separated: bool = True) -> None:
        if content.strip():
            self.open_block("assistant", separated=separated)
            self.console.print(CompactMarkdown(content))

    def print_tools(self, tools: list[dict[str, Any]]) -> None:
        rows = []
        for tool in tools:
            required = tool.get("parameters", {}).get("required", [])
            suffix = f" · required: {', '.join(required)}" if required else ""
            rows.append((str(tool.get("name", "")), str(tool.get("description", "")) + suffix))
        self._print_menu_rows("Jasmine Tools", rows)

    def print_menu(self) -> None:
        self._print_menu_rows("Menu", [(command.strip(), desc) for command, desc in COMMAND_ROWS])

    def _print_menu_rows(self, title: str, rows: list[tuple[str, str]]) -> None:
        self.open_block("menu", separated=False)
        header = Text()
        header.append("• ", style=SUCCESS_STYLE)
        header.append(title, style="bold #cccccc")
        self.console.print(header)
        # Dynamic label width: at least 12, but stretch to fit the widest label
        max_label = max((len(label) for label, _ in rows), default=12)
        label_width = max(12, min(max_label + 2, 24))
        term_width = shutil.get_terminal_size().columns
        desc_width = max(20, term_width - label_width - 5)  # 5 = indent "  └ " / "    "
        for index, (label, description) in enumerate(rows):
            label_fmt = str(label).ljust(label_width)
            desc = self._short(description, desc_width)
            line = Text()
            line.append("  └ " if index == 0 else "    ", style="#737373")
            line.append(label_fmt, style="bold #e5e7eb")
            line.append(desc, style="#b8b8b8")
            self.console.print(line)

    def print_usage(self, usage: dict[str, Any]) -> None:
        self.open_block("usage", separated=False)
        line = Text()
        line.append("• ", style=SUCCESS_STYLE)
        line.append(
            "usage"
            f" · provider={usage.get('provider', 'unknown')}"
            f" · requests={usage.get('requests', 0)}"
            f" · total={usage.get('total_tokens', 0)}"
            f" · prompt={usage.get('prompt_tokens', 0)}"
            f" · completion={usage.get('completion_tokens', 0)}"
            f" · reasoning={usage.get('reasoning_tokens', 0)}"
            f" · cache_hit={usage.get('prompt_cache_hit_tokens', 0)}"
            f" · cost=${float(usage.get('cost_usd', 0.0) or 0.0):.6f}",
            style="bold #b8b8b8",
        )
        self.console.print(line)

    @asynccontextmanager
    async def assistant_stream(self) -> AsyncIterator[AssistantStream]:
        stream = AssistantStream(self)
        async with stream:
            yield stream

    @asynccontextmanager
    async def tool_activity(self, name: str, args: dict[str, Any]) -> AsyncIterator[ToolActivity]:
        activity = ToolActivity(self, name, args)
        async with activity:
            yield activity

    def _tool_activity_renderable(self, name: str, args: dict[str, Any]) -> RenderableType:
        if name == "exec_command":
            return Text.assemble(
                Text("Running ", style="bold #d1d1d1"),
                Text(self._short(str(args.get("command") or args.get("cmd") or ""), 104), style="bold #d1d1d1"),
            )
        row = Text()
        row.append("Running ", style="bold #d1d1d1")
        row.append(self._short(self._tool_label(name, args), 104), style="bold #d1d1d1")
        return row

    def _lexer_for_path(self, path: str | None, code: str = "") -> str:
        """Return a Pygments/Rich lexer name for a file path.

        The explicit map covers common config files and extension-less scripts;
        Pygments then handles the long tail of languages via filename guessing.
        """
        name = (path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        ext = Path(name).suffix.lower()
        by_name = {
            "dockerfile": "docker",
            "containerfile": "docker",
            "makefile": "make",
            "justfile": "make",
            "rakefile": "ruby",
            "gemfile": "ruby",
            "podfile": "ruby",
            "vagrantfile": "ruby",
            "jenkinsfile": "groovy",
            "cmakelists.txt": "cmake",
            "package.json": "json",
            "tsconfig.json": "json",
            "composer.json": "json",
            "cargo.toml": "toml",
            "pyproject.toml": "toml",
            ".env": "bash",
            ".gitignore": "gitignore",
            ".dockerignore": "gitignore",
        }
        by_ext = {
            ".py": "python", ".pyw": "python", ".pyi": "python",
            ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "tsx",
            ".html": "html", ".htm": "html", ".xml": "xml", ".svg": "xml",
            ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
            ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
            ".sh": "bash", ".bash": "bash", ".zsh": "zsh", ".fish": "fish", ".ps1": "powershell",
            ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
            ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".groovy": "groovy",
            ".cs": "csharp", ".fs": "fsharp", ".fsx": "fsharp",
            ".php": "php", ".rb": "ruby", ".lua": "lua", ".pl": "perl", ".pm": "perl",
            ".swift": "swift", ".dart": "dart", ".r": "r", ".jl": "julia", ".ex": "elixir", ".exs": "elixir",
            ".erl": "erlang", ".hrl": "erlang", ".clj": "clojure", ".cljs": "clojure", ".hs": "haskell",
            ".sql": "sql", ".graphql": "graphql", ".gql": "graphql",
            ".md": "markdown", ".markdown": "markdown", ".rst": "rst", ".tex": "tex",
            ".vue": "vue", ".svelte": "html", ".astro": "html",
            ".diff": "diff", ".patch": "diff",
            ".proto": "protobuf", ".tf": "terraform", ".hcl": "hcl",
            ".sol": "solidity", ".nim": "nim", ".zig": "zig",
        }
        if name in by_name:
            return by_name[name]
        if ext in by_ext:
            return by_ext[ext]
        if path:
            try:
                guessed = Syntax.guess_lexer(path, code or " ")
                aliases = getattr(guessed, "aliases", None)
                if aliases:
                    return str(aliases[0])
                lexer_name = getattr(guessed, "name", None)
                if lexer_name:
                    return str(lexer_name).lower()
            except Exception:
                pass
        return "text"

    def _syntax_highlight_line(self, line: str, path: str | None, bg: str, *, dim: bool = False) -> Text:
        lexer = self._lexer_for_path(path, line)
        try:
            highlighted = Syntax("", lexer, theme="ansi_dark", background_color=bg).highlight(line)
            if highlighted.plain.endswith("\n"):
                highlighted.right_crop(1)
        except Exception:
            highlighted = Text(line, style="#d1d5db")
        if dim:
            highlighted.stylize("dim")
        if len(highlighted):
            highlighted.stylize(f"on {bg}", 0, len(highlighted))
        return highlighted

    def _bash_highlight_line(self, line: str, *, dim: bool = False) -> Text:
        try:
            highlighted = Syntax(line, "bash", theme="ansi_dark").highlight(line)
            if highlighted.plain.endswith("\n"):
                highlighted.right_crop(1)
        except Exception:
            highlighted = Text(line, style="#d1d5db")
        if dim:
            highlighted.stylize("dim")
        return highlighted
    def _tool_label(self, name: str, args: dict[str, Any]) -> str:
        if name == "exec_command":
            return str(args.get("command") or args.get("cmd") or "")
        if name == "write_stdin":
            target = args.get("session_name") or args.get("session_id") or "session"
            return f"write_stdin · {target}"
        if name == "ask_user":
            return "ask_user"
        if name == "web_search":
            return f"web_search · {str(args.get('query', ''))[:80]}"
        if name == "web_extract":
            return f"web_extract · {str(args.get('url', ''))[:80]}"
        if name == "apply_patch":
            return "apply_patch"
        if name == "multi_tool_use_parallel":
            return f"exploring · {len(args.get('tool_uses', []) or [])} operations"
        if name == "update_plan":
            return "update_plan"
        if name == "view_image":
            return f"view_image · {args.get('path', '')}"
        if name == "list_skills":
            return "list_skills"
        if name == "read_skill":
            return f"read_skill · {args.get('name', '')}"
        if name == "terminal_screen":
            return f"terminal_screen · {args.get('session_name') or args.get('session_id') or 'session'}"
        if name == "terminal_close":
            return f"terminal_close · {args.get('session_name') or args.get('session_id') or 'session'}"
        return name

    def print_tool_result(self, name: str, result: dict[str, Any]) -> None:
        if result.get("cached_by_agent"):
            return
        if name == "update_plan" and result.get("ok") is not False:
            return
        self.open_block("tool")
        ok = result.get("ok") is not False
        if name == "exec_command":
            self.console.print(self._command_renderable("Ran", str(result.get("command", "")), ok=ok))
            preview = str(result.get("output") or result.get("error") or "")
            if result.get("hint"):
                preview += ("\n" if preview else "") + "hint: " + str(result["hint"])
            if preview:
                self.console.print(self._tree_renderable(preview, ok=ok))
            return
        if name == "apply_patch":
            diff = str(result.get("diff", ""))
            if not ok:
                self._print_tool_header("Failed apply_patch", ok=False)
                self.console.print(self._tree_renderable(str(result.get("error") or result), ok=False))
                return
            added, removed = self._diff_stats(diff)
            paths = ", ".join(str(path) for path in result.get("paths", []) or [])
            label = f"Edited {paths or result.get('path') or 'files'}"
            if added or removed:
                label += f" (+{added} -{removed})"
            self._print_tool_header(label, ok=ok)
            if diff:
                self.print_diff(diff, already_open=True, show_header=False)
            return
        if name == "multi_tool_use_parallel":
            self._print_tool_header("Explored", ok=ok)
            lines = self._parallel_exploration_lines(result)
            if lines:
                self.console.print(self._tree_renderable("\n".join(lines), ok=ok))
            return
        if name == "write_stdin":
            self._print_tool_header(f"Wrote stdin to session {result.get('session_id', '')}", ok=ok)
        elif name == "ask_user":
            answer = str(result.get("answer", ""))
            self._print_tool_header(f"Asked: {str(result.get('question', ''))[:80]}", ok=ok)
            if answer:
                self.console.print(self._tree_renderable(f"Answer: {answer}", ok=ok))
        elif name == "web_search":
            count = result.get("count", 0)
            self._print_tool_header(f"Searched web for '{str(result.get('query', ''))[:60]}' ({count} results)", ok=ok)
        elif name == "web_extract":
            title = str(result.get("title", ""))
            text_len = result.get("text_length", 0)
            links_count = len(result.get("links", []) or [])
            images_count = len(result.get("images", []) or [])
            self._print_tool_header(
                f"Extracted: {title[:80] or result.get('url', '')} | {text_len} chars, {links_count} links, {images_count} imgs",
                ok=ok,
            )
        elif name == "view_image":
            self._print_tool_header(f"Viewed {result.get('path', '')}", ok=ok)
        elif name == "list_skills":
            self._print_tool_header("Listed skills", ok=ok)
        elif name == "read_skill":
            self._print_tool_header(f"Read skill {result.get('name', '')}", ok=ok)
        elif name == "terminal_screen":
            self._print_tool_header(f"Read terminal {result.get('session_name') or result.get('session_id', '')}", ok=ok)
        elif name == "terminal_close":
            self._print_tool_header(f"Closed terminal {result.get('session_name') or result.get('session_id', '')}", ok=ok)
        else:
            self._print_tool_header(("Completed " if ok else "Failed ") + name, ok=ok)
        preview = self._result_preview(name, result, ok)
        if preview:
            self.console.print(self._tree_renderable(preview, ok=ok))

    def _print_tool_header(self, label: str, *, ok: bool = True) -> None:
        header = Text()
        header.append("• ", style=SUCCESS_STYLE if ok else "#ef4444 bold")
        header.append(label, style="bold #cccccc" if ok else "bold #fecaca")
        self.console.print(header)

    def _command_renderable(self, verb: str, command: str, *, ok: bool = True) -> RenderableType:
        lines = command.splitlines() or [""]
        rendered: list[Text] = []
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append(f"• {verb} ", style=SUCCESS_STYLE if ok else "#ef4444 bold")
        header.append_text(self._bash_highlight_line(self._short(lines[0], 112)))
        rendered.append(header)
        shown = lines[1:4]
        for line in shown:
            line_text = Text(no_wrap=True, overflow="ellipsis")
            line_text.append("  │ ", style="#737373")
            line_text.append_text(self._bash_highlight_line(self._short(line, 110)))
            rendered.append(line_text)
        omitted = len(lines) - len(shown) - 1
        if omitted > 0:
            rendered.append(Text(f"  │ … +{omitted} lines", style="#8a8a8a italic", no_wrap=True, overflow="ellipsis"))
        return Group(*rendered) if len(rendered) > 1 else rendered[0]

    def _tree_renderable(self, text: str, *, ok: bool = True, max_lines: int = 14) -> RenderableType:
        raw_lines = text.strip("\r\n").splitlines() or [""]
        shown = raw_lines[:max_lines]
        omitted = len(raw_lines) - len(shown)
        if omitted:
            shown.append(f"… +{omitted} lines omitted")
        base_style = "bold #a3a3a3 on #1e1e1e" if ok else "bold #fecaca on #1e1e1e"
        rendered = [
            Text(
                ("  └ " if index == 0 else "    ") + self._short(line, 116),
                style=base_style,
                no_wrap=True,
                overflow="ellipsis",
            )
            for index, line in enumerate(shown)
        ]
        return Group(*rendered)

    def _diff_stats(self, diff: str) -> tuple[int, int]:
        lines = diff.splitlines()
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        return added, removed

    def _parallel_exploration_lines(self, result: dict[str, Any]) -> list[str]:
        grouped: dict[str, list[str]] = {}
        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            if name == "view_image":
                detail = self._display_path(str(args.get("path", "")))
                if detail and detail not in grouped.setdefault("Read", []):
                    grouped["Read"].append(detail)
                continue
            command = str(args.get("command") or args.get("cmd") or "").strip()
            action, detail = self._describe_exploration(command)
            if detail and detail not in grouped.setdefault(action, []):
                grouped[action].append(detail)
        lines = []
        for action, details in grouped.items():
            if action == "Run":
                lines.extend(f"Run {detail}" for detail in details)
            else:
                lines.append(f"{action} {', '.join(details)}")
        return lines

    def _describe_exploration(self, command: str) -> tuple[str, str]:
        try:
            parts = shlex.split(command)
        except ValueError:
            return "Run", self._short(command, 108)
        if not parts:
            return "Run", command
        executable = Path(parts[0]).name
        if executable in {"cat", "head", "tail"}:
            paths = [part for part in parts[1:] if not part.startswith("-")]
            return "Read", self._display_path(paths[-1] if paths else command)
        if executable == "sed":
            return "Read", self._display_path(parts[-1] if len(parts) > 1 else command)
        if executable in {"rg", "grep"} and "--files" not in parts:
            pattern, raw_targets = self._search_pattern_and_targets(parts[1:])
            targets = ", ".join(self._display_path(part) for part in raw_targets) or "."
            return "Search", f"{pattern} in {targets}"
        if executable in {"ls", "find", "tree"} or (executable == "rg" and "--files" in parts):
            values = [part for part in parts[1:] if not part.startswith("-")]
            return "List", ", ".join(self._display_path(part) for part in values) or "."
        if executable == "git" and len(parts) > 1:
            return "Read", "git " + " ".join(parts[1:3])
        return "Run", self._short(command, 108)

    def _search_pattern_and_targets(self, args: list[str]) -> tuple[str, list[str]]:
        patterns: list[str] = []
        values: list[str] = []
        options_with_value = {
            "-A", "-B", "-C", "-e", "-f", "-g", "-m", "-t",
            "--after-context", "--before-context", "--context", "--encoding",
            "--engine", "--file", "--glob", "--max-count", "--regexp", "--sort",
            "--sortr", "--type", "--type-add",
        }
        index = 0
        while index < len(args):
            value = args[index]
            if value in options_with_value:
                if index + 1 < len(args) and value in {"-e", "--regexp"}:
                    patterns.append(args[index + 1])
                index += 2
                continue
            if value.startswith(("--regexp=",)):
                patterns.append(value.split("=", 1)[1])
                index += 1
                continue
            if value.startswith("-"):
                index += 1
                continue
            values.append(value)
            index += 1
        if not patterns and values:
            patterns.append(values.pop(0))
        return "|".join(patterns) or "pattern", values

    def _display_path(self, path: str) -> str:
        clean = path.removeprefix("./").rstrip("/")
        if "/" in clean:
            return Path(clean).name or clean
        return clean or "."

    def _result_preview(self, name: str, result: dict[str, Any], ok: bool) -> str:
        if result.get("cached_by_agent"):
            return str(result.get("note", "same tool request already satisfied"))
        if result.get("denied"):
            return str(result.get("error", "denied"))
        if result.get("ok") is False:
            return self._truncate(str(result.get("error") or result.get("output") or result), 3500, mode="tail")
        if name in {"exec_command", "write_stdin", "terminal_screen", "terminal_close"}:
            command = str(result.get("command", ""))
            output = str(result.get("output", ""))
            parts = []
            if command:
                parts.append(f"$ {command}")
            if output:
                parts.append(self._truncate_lines(output, max_lines=80, max_chars=30000))
            if result.get("hint"):
                parts.append("hint: " + str(result["hint"]))
            if result.get("job") and not output:
                parts.append(json.dumps(result.get("job"), ensure_ascii=False, indent=2))
            return "\n".join(parts)
        if name == "view_image":
            return json.dumps({k:v for k,v in result.items() if k != "base64"}, ensure_ascii=False, indent=2)
        if name == "list_skills":
            return str(result.get("output", ""))
        if name == "read_skill":
            return str(result.get("path", ""))
        if name == "multi_tool_use_parallel":
            lines = []
            for item in result.get("results", []) or []:
                nested = item.get("result") or {}
                command = (item.get("args") or {}).get("command")
                if command:
                    lines.append(f"$ {command}  [exit {nested.get('returncode')}]")
            return "\n".join(lines)
        if name == "web_extract":
            parts = []
            title = str(result.get("title", ""))
            if title:
                parts.append(f"Title: {title}")
            desc = str(result.get("description", ""))
            if desc:
                parts.append(f"Description: {desc}")
            links = result.get("links", []) or []
            if links:
                parts.append(f"Links ({len(links)}):")
                for link in links[:6]:
                    parts.append(f"  [{link.get('text', '')[:50] or '(no text)'}]({link.get('href', '')})")
                if len(links) > 6:
                    parts.append(f"  ... +{len(links) - 6} more")
            images = result.get("images", []) or []
            if images:
                parts.append(f"Images: {len(images)}")
            headings = result.get("headings", []) or []
            if headings:
                parts.append(f"Headings ({len(headings)}):")
                for h in headings[:5]:
                    parts.append(f"  H{h['level']}: {h['text'][:60]}")
            return "\n".join(parts)
        return ""

    def print_diff(self, diff: str, already_open: bool = False, show_header: bool = True) -> None:
        if not diff.strip():
            return
        if not already_open:
            self.open_block("diff")
        visible = self._truncate(diff, 42000, mode="middle")
        width = max(40, self.console.width)
        # Compute stat summary
        added = sum(1 for l in visible.splitlines() if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in visible.splitlines() if l.startswith("-") and not l.startswith("---"))
        # Extract file path for header
        current_path: str | None = None
        for raw in visible.splitlines():
            if raw.startswith("+++ "):
                c = raw[4:].strip()
                if c.startswith("b/"):
                    c = c[2:]
                if c and c != "/dev/null":
                    current_path = c
                break
        path_label = f"  {current_path}" if current_path else ""
        stat_label = f"  +{added} / -{removed}" if added or removed else ""
        if show_header:
            header = Text(f"diff{path_label}{stat_label}", style="bold #86efac")
            self.console.print(header)
        # Render with line number tracking and word-level diff
        lines = visible.rstrip().splitlines()
        line_num: int | None = None
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.replace("\t", "    ")
            if line.startswith("@@ "):
                m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if m:
                    line_num = int(m.group(2))
                self.console.print(self._diff_line_text(line, current_path, width), soft_wrap=False)
                i += 1
                continue
            if line.startswith(("---", "+++")):
                if line.startswith("+++ "):
                    c = line[4:].strip()
                    if c.startswith("b/"):
                        c = c[2:]
                    if c and c != "/dev/null":
                        current_path = c
                self.console.print(self._diff_line_text(line, current_path, width), soft_wrap=False)
                i += 1
                continue
            if line.startswith("-") and not line.startswith("---"):
                self.console.print(self._diff_line_text(line, current_path, width, line_num=line_num), soft_wrap=False)
                i += 1
                continue
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(self._diff_line_text(line, current_path, width, line_num=line_num), soft_wrap=False)
                if line_num is not None:
                    line_num += 1
                i += 1
                continue
            # Context or other line
            self.console.print(self._diff_line_text(line, current_path, width, line_num=line_num), soft_wrap=False)
            if line.startswith(" ") and line_num is not None:
                line_num += 1
            elif not line.startswith(" ") and not line.startswith("\\"):
                line_num = None
            i += 1

    def _diff_line_text(self, line: str, path: str | None, width: int, line_num: int | None = None) -> Text:
        if len(line) > width - 2:
            line = line[: width - 3] + "…"
        if line.startswith("+") and not line.startswith("+++"):
            return self._diff_code_line("+", line[1:], path, width, bg="#123524", marker_style="#86efac bold", dim=False, line_num=line_num)
        if line.startswith("-") and not line.startswith("---"):
            return self._diff_code_line("-", line[1:], path, width, bg="#3a1717", marker_style="#fca5a5 bold", dim=False, line_num=line_num)
        if line.startswith(" "):
            return self._diff_code_line(" ", line[1:], path, width, bg="#202020", marker_style="#8c8c8c", dim=True, line_num=line_num)
        if line.startswith("@@"):
            return Text((" " + line).ljust(width), style="#ffffff on #2c2c2c bold")
        if line.startswith(("---", "+++")):
            return Text((" " + line).ljust(width), style="#e5e7eb on #2c2c2c bold")
        return Text((" " + line).ljust(width), style="#b8b8b8 on #202020")

    def _diff_code_line(self, marker: str, payload: str, path: str | None, width: int, *, bg: str, marker_style: str, dim: bool, line_num: int | None = None) -> Text:
        out = Text()
        out.append(" ", style=f"on {bg}")
        out.append(marker, style=f"{marker_style} on {bg}")
        # Line number gutter
        if line_num is not None:
            num_str = f"{line_num:>4} "
            out.append(num_str, style=f"#5a5a5a on {bg}")
        else:
            out.append("     ", style=f"on {bg}")
        gutter_width = 7  # 2 (marker) + 5 (gutter)
        code_width = max(1, width - gutter_width)
        if len(payload) > code_width:
            payload = payload[: max(1, code_width - 1)] + "…"
        highlighted = self._syntax_highlight_line(payload, path, bg, dim=dim)
        out.append_text(highlighted)
        if out.cell_len < width:
            out.append(" " * (width - out.cell_len), style=f"on {bg}")
        return out

    def _print_labeled_block(self, label: str, text: str, fg: str, bg: str, label_style: str, accent: str, wrap: bool, max_lines: int | None) -> None:
        self.console.print(Text(f" {label} ", style=label_style))
        self._print_block(text, fg=fg, bg=bg, wrap=wrap, max_lines=max_lines, accent=accent)
    def _print_block(self, text: str, fg: str = "#b2b2b2", bg: str = "#303030", wrap: bool = False, max_lines: int | None = None, accent: str | None = None) -> None:
        term_width = shutil.get_terminal_size().columns
        width = max(40, term_width)
        lines = self._prepare_lines(text, width - 2, wrap=wrap, max_lines=max_lines)
        for line in lines:
            prefix = "▌" if accent else " "
            prefix_style = accent or bg
            out = Text()
            out.append(prefix, style=f"{prefix_style} on {bg}")
            body_text = Text(" " + line, style=f"{fg} on {bg}")
            padding_needed = width - 1 - body_text.cell_len
            if padding_needed > 0:
                body_text.append(" " * padding_needed, style=f"{fg} on {bg}")
            out.append_text(body_text)
            self.console.print(out, soft_wrap=False)

    def _prepare_lines(self, text: str, width: int, wrap: bool, max_lines: int | None) -> list[str]:
        raw_lines = text.rstrip("\n").splitlines() or [""]
        lines: list[str] = []
        for raw in raw_lines:
            clean = raw.replace("\t", "    ")
            if wrap:
                lines.extend(textwrap.wrap(clean, width=max(20, width), replace_whitespace=False, drop_whitespace=False) or [""])
            else:
                lines.append(clean if len(clean) <= width else clean[: width - 1] + "…")
        if max_lines is not None and len(lines) > max_lines:
            omitted = len(lines) - max_lines
            keep_head = max(1, max_lines // 2)
            keep_tail = max(0, max_lines - keep_head - 1)
            if keep_tail > 0:
                lines = lines[:keep_head] + [f"[truncated: {omitted} more line(s)]"] + lines[-keep_tail:]
            else:
                lines = lines[:keep_head] + [f"[truncated: {omitted} more line(s)]"]
        return lines

    def update_plan(self, items: list[dict[str, Any]], show: bool = True) -> bool:
        next_items = [PlanItem(text=str(item.get("text", "")).strip(), status=str(item.get("status", "pending"))) for item in items]
        next_items = [item for item in next_items if item.text]
        signature = json.dumps([item.__dict__ for item in next_items], ensure_ascii=False, sort_keys=True)
        if signature == self._last_plan_signature:
            self.plan_items = next_items
            return False
        self.plan_items = next_items
        self._last_plan_signature = signature
        if show:
            self.open_block("plan", separated=False)
            header = Text()
            header.append("• ", style=SUCCESS_STYLE)
            header.append("Updated Plan", style="bold #cccccc")
            self.console.print(header)
            for index, item in enumerate(self.plan_items[:12]):
                icon, style = self._plan_style(item.status)
                prefix = "  └ " if index == 0 else "    "
                line = Text()
                line.append(prefix, style="#737373")
                line.append(icon + " ", style=style)
                line.append(item.text, style=style)
                self.console.print(line)
        return True

    def _plan_style(self, status: str) -> tuple[str, str]:
        status = status.lower().strip()
        if status in {"done", "completed", "checked"}:
            return "✔", "#6b6b6b strike"
        if status in {"in_progress", "running", "active", "doing"}:
            return "▣", "bold #22c55e"
        if status in {"blocked", "failed", "error"}:
            return "!", "#fca5a5"
        return "□", "#737373"

    def command_requires_approval(self, command: str) -> bool:
        command = command.strip()
        if not command:
            return False
        if self._starts_with_sudo(command) or self._looks_destructive(command):
            return True
        if self._is_safe_auto_command(command):
            return False
        return not any(command.startswith(prefix) for prefix in self.trusted_prefixes)

    def is_safe_auto_command(self, command: str) -> bool:
        return self._is_safe_auto_command(command.strip())

    def _strip_leading_cd(self, command: str) -> str:
        # Accept the common model pattern `cd some/path && <read-only command>`
        # without approving every focused search manually. Other chaining stays
        # approval-gated.
        match = re.match(r"^\s*cd\s+(?:'[^']+'|\"[^\"]+\"|[^&;|]+)\s*&&\s*(.+?)\s*$", command)
        return match.group(1) if match else command

    def _is_safe_auto_command(self, command: str) -> bool:
        command = self._strip_leading_cd(command.strip())
        if any(token in command for token in (";", "&&", "||", "|", ">", "<", "`", "$(", "\n", "&", ">>", "2>")):
            return False
        try:
            parts = shlex.split(command)
        except Exception:
            return False
        if not parts:
            return False
        name = parts[0]
        # Path-only manipulators: safe regardless of argument paths
        if name in {"dirname", "basename", "realpath", "readlink"}:
            return True
        if self._has_sensitive_or_external_path(parts[1:]):
            return False
        if name in {"ls", "pwd", "rg", "grep", "cat", "head", "tail", "wc", "file", "tree", "echo", "printf", "which", "sort", "uniq", "cut", "tr", "awk", "env", "printenv"}:
            return True
        if name == "sed":
            return not any(part == "-i" or part.startswith("-i") for part in parts[1:])
        if name == "find":
            return not any(part in {"-delete", "-exec", "-execdir"} for part in parts[1:])
        if name == "git" and len(parts) >= 2:
            return parts[1] in {"status", "diff", "log", "show", "ls-files", "grep", "branch", "rev-parse"}
        return False

    def _has_sensitive_or_external_path(self, args: list[str]) -> bool:
        sensitive = (".env", ".ssh", ".aws", "id_rsa", "id_ed25519", "credentials", "secrets")
        for arg in args:
            lowered = arg.lower()
            if any(token in lowered for token in sensitive):
                return True
            if arg.startswith(("~", "/")) or arg == ".." or arg.startswith("../") or "/../" in arg:
                return True
        return False

    async def approve_command(self, command: str, cwd: str | None = None, reason: str = "", force: bool = False) -> tuple[bool, bool]:
        command = command.strip()
        if not force and not self.command_requires_approval(command):
            return True, False
        self.open_block("approval")
        self.console.print(Text("  Would you like to run the following command?", style="#e5e7eb"))
        if cwd and cwd != ".":
            self.console.print(Text(f"  Workdir: {cwd}", style="#a3a3a3"))
        if reason:
            self.console.print(Text(f"  Reason: {reason}", style="#a3a3a3"))
        elif self._looks_destructive(command):
            self.console.print(Text("  Reason: high-risk command; review carefully.", style="#fca5a5"))
        self.console.print()
        self.console.print(Text(f"  $ {command}", style="#f3f4f6"))
        self.console.print()
        self.console.print(Text("› 1. Yes, proceed (allow)", style="#e5e7eb"))
        self.console.print(Text("  2. Yes, and don't ask again for a safe prefix (remember)", style="#d1d5db"))
        self.console.print(Text("  3. No, and tell Jasmine what to do differently (deny)", style="#d1d5db"))
        if not sys.stdin.isatty():
            self.console.print(Text("  Non-interactive input cannot approve this command.", style="#fca5a5"))
            return False, False
        self._aux_prompt_depth += 1
        try:
            while True:
                with patch_stdout(raw=True):
                    answer = await self.approval_session.prompt_async(
                        [("class:approval", "approve › ")],
                        pre_run=lambda: self.approval_session.default_buffer.start_completion(select_first=True),
                    )
                answer = (answer or "allow").strip().lower()
                if answer.startswith(("1", "a", "y")):
                    self.console.print(Text(f"✔ You approved Jasmine to run {self._short(command, 96)} this time", style=SUCCESS_STYLE))
                    return True, False
                if answer.startswith(("3", "d", "n")):
                    return False, False
                if answer.startswith(("2", "r", "p")):
                    if self._starts_with_sudo(command):
                        self.console.print("[red]sudo commands cannot be remembered.[/red]")
                        continue
                    default_prefix = self._default_prefix(command)
                    with patch_stdout(raw=True):
                        prefix = await self.line_session.prompt_async([("class:approval", f"prefix [{default_prefix}] › ")], default=default_prefix)
                    prefix = prefix.strip() or default_prefix
                    if prefix and prefix not in self.trusted_prefixes:
                        self.trusted_prefixes.append(prefix)
                        self.trusted_prefixes.sort(key=len, reverse=True)
                        self._save_trusted_prefixes()
                        self.console.print(Text(f"✔ You approved Jasmine to run commands starting with {prefix}", style=SUCCESS_STYLE))
                    return True, True
        finally:
            self._aux_prompt_depth -= 1
            self._reinstall_interrupt_handler()

    def _starts_with_sudo(self, command: str) -> bool:
        try:
            first = shlex.split(command)[0]
        except Exception:
            first = command.split(maxsplit=1)[0] if command.split() else ""
        return first == "sudo"

    def _looks_destructive(self, command: str) -> bool:
        lowered = command.lower()
        padded = " " + lowered + " "
        risky = [" rm ", "rm -", "sudo", "mkfs", "dd if=", ":(){", "chmod -r", "chown -r", "git reset --hard", "git clean -fd"]
        return any(token in padded for token in risky)

    def _default_prefix(self, command: str) -> str:
        try:
            parts = shlex.split(command)
        except Exception:
            parts = command.split()
        if not parts:
            return command
        if parts[0] in {"python", "python3", "node", "npm", "pnpm", "yarn", "pytest", "ruff", "mypy", "git", "make"}:
            return " ".join(parts[:2]) if len(parts) > 1 else parts[0]
        return parts[0]

    def print_trusted(self) -> None:
        self.open_block("menu", separated=False)
        if not self.trusted_prefixes:
            self.console.print("[#949494]none[/#949494]")
            return
        for prefix in self.trusted_prefixes:
            self.console.print(f"[green]●[/] [#b2b2b2]{prefix}[/#b2b2b2]")

    def _short(self, text: str, max_len: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _truncate_lines(self, text: str, max_lines: int, max_chars: int) -> str:
        if not text:
            return ""
        lines = text.splitlines()
        shown = lines[:max_lines]
        body = "\n".join(shown)
        omitted_lines = max(0, len(lines) - len(shown))
        body = self._truncate(body, max_chars, mode="tail")
        if omitted_lines:
            body += f"\n[truncated: {omitted_lines} more line(s)]"
        return body

    def _truncate(self, text: str, max_chars: int, mode: str = "tail") -> str:
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        marker = f"[truncated: {omitted} chars omitted]"
        if mode == "head":
            return text[:max_chars] + "\n" + marker
        if mode == "middle" and max_chars > len(marker) + 20:
            keep = max_chars - len(marker) - 2
            left = keep // 2
            right = keep - left
            return text[:left] + "\n" + marker + "\n" + text[-right:]
        return marker + "\n" + text[-max_chars:]
