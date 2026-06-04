from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import Agent
from .ai import load_backend
from . import __version__
from .config import JasmineConfig, get_theme
from .conversation import ConversationStore
from .providers import ProviderConfig
from .tools import ToolRegistry, Tool
from .ui import TerminalUI


PASTE_COMPACT_THRESHOLD = 0
STATE_PATH = Path.home() / ".jasmine_cli_state.json"


def _load_last_workspace() -> str | None:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        path = data.get("last_workspace")
        if path and Path(path).expanduser().is_dir():
            return str(path)
    except Exception:
        pass
    return None


def _save_last_workspace(root: Path) -> None:
    try:
        STATE_PATH.write_text(json.dumps({"last_workspace": str(root)}, indent=2), encoding="utf-8")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jasmine/TUI")
    parser.add_argument("workspace", nargs="?", default=None, help="Workspace root directory")
    parser.add_argument("--backend", default=os.environ.get("JASMINE_BACKEND") or os.environ.get("AI_CODE_BACKEND", "deepseek"), help="Backend spec: deepseek, mock, or module:ClassName")
    parser.add_argument("--model", default=None, help="Override JASMINE_MODEL for the real backend")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "max", "xhigh"], default=None, help="Override provider reasoning effort")
    parser.add_argument("--thinking", choices=["on", "off"], default=None, help="Enable or disable provider-side thinking when the backend supports it")
    parser.add_argument("--fast", action="store_true", help="Shortcut: high reasoning effort with provider thinking on")
    parser.add_argument("--quality", action="store_true", help="Shortcut: xhigh reasoning effort with provider thinking on")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--paste-threshold", type=int, default=int(os.environ.get("JASMINE_PASTE_THRESHOLD") or os.environ.get("AI_CODE_PASTE_THRESHOLD", PASTE_COMPACT_THRESHOLD)), help="Compact user messages above this character count. 0 disables compaction.")
    return parser.parse_args()


def compact_large_paste(root: Path, text: str, threshold: int) -> tuple[str, str | None, int]:
    if threshold <= 0 or len(text) <= threshold:
        return text, None, len(text)
    paste_dir = root / ".jasmine" / "pastes"
    paste_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = paste_dir / f"paste_{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    rel = str(path.relative_to(root))
    compact = (
        f"[pasted-content: {len(text)} characters]\n"
        f"Full pasted content saved at `{rel}`. Inspect it with exec_command only if the task needs it."
    )
    return compact, rel, len(text)


def register_agent_tools(tools: ToolRegistry) -> None:
    tools.register(
        Tool(
            name="multi_tool_use_parallel",
            description="Batch 1-4 independent read-only exec_command, view_image, web_search, or web_extract operations into one model round. Use this when inspecting unrelated files, images, or web pages.",
            schema={
                "type": "object",
                "properties": {
                    "tool_uses": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "recipient_name": {"type": "string", "enum": ["exec_command", "view_image", "web_search", "web_extract"]},
                                "parameters": {"type": "object"},
                            },
                            "required": ["recipient_name", "parameters"],
                        },
                    }
                },
                "required": ["tool_uses"],
            },
            fn=lambda _args: None,
        )
    )
    tools.register(
        Tool(
            name="update_plan",
            description="Create or update the visible live plan. Use for every multi-step task and update statuses as work progresses.",
            schema={
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "blocked", "failed"]},
                            },
                            "required": ["step", "status"],
                        },
                    }
                },
                "required": ["plan"],
            },
            fn=lambda _args: None,
        )
    )
    tools.register(
        Tool(
            name="ask_user",
            description="Ask the user a question and wait for their response. Use this when you need clarification, a decision, or specific input from the user to proceed. The question will be displayed in green to differentiate it from other output.",
            schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to display to the user"},
                },
                "required": ["question"],
            },
            fn=lambda _args: None,
        )
    )
    tools.register(
        Tool(
            name="web_search",
            description="Search the web using DuckDuckGo and return a list of results with title, URL, and description. Use this to find current information, documentation, or answers from the internet.",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results (1-20)"},
                    "region": {"type": "string", "default": "wt-wt", "description": "Region code, e.g. us-en, fr-fr, wt-wt"},
                },
                "required": ["query"],
            },
            fn=tools.web_search,
        )
    )
    tools.register(
        Tool(
            name="web_extract",
            description="Fetch and extract structured content from a web page. Returns JSON with title, description, text (readability-cleaned), headings structure (h1-h6), all links with hrefs, images with src/alt, meta tags, JSON-LD structured data, and SSR payloads (__NEXT_DATA__, __NUXT__, __INITIAL_STATE__). Use after web_search to read the full content of a promising result.",
            schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the page to extract content from"},
                    "max_chars": {"type": "integer", "default": 5000, "description": "Maximum characters to return (500-30000)"},
                },
                "required": ["url"],
            },
            fn=tools.web_extract,
        )
    )


async def handle_command(user_text: str, agent: Agent, ui: TerminalUI, tools_ref: dict[str, ToolRegistry], root_ref: dict[str, Path], conv_store: ConversationStore) -> bool:
    lower = user_text.lower().strip()
    if lower in {"/exit", "/quit", "exit", "quit"}:
        raise EOFError
    if lower in {"/menu", "/"}:
        ui.print_menu()
        return True
    if lower == "/tools":
        ui.print_tools(tools_ref["tools"].schemas())
        return True
    if lower == "/trusted":
        ui.print_trusted()
        return True
    if lower == "/usage":
        agent.show_usage()
        return True
    if lower == "/provider":
        await handle_provider(agent, ui, root_ref["root"])
        return True
    if lower == "/clear":
        agent.clear_context(announce=False)
        conv_store.start_conversation()
        ui.clear_input_history()
        ui.redraw_transcript(str(root_ref["root"]), agent.messages)
        ui.console.print("[green]•[/] [#9e9e9e]context cleared[/#9e9e9e]")
        ui.console.print()
        return True
    if lower == "/resume":
        await handle_resume(agent, ui, conv_store, root_ref)
        return True
    return False


async def handle_resume(agent: Agent, ui: TerminalUI, conv_store: ConversationStore, root_ref: dict[str, Path]) -> None:
    """List past conversations and let the user resume one."""
    conversations = conv_store.list_conversations()
    if not conversations:
        ui.console.print("[#9e9e9e]no past conversations to resume[/#9e9e9e]")
        return

    ui.print_conversation_list(conversations)
    # Ask which one to resume
    ui.console.print()
    ui.console.print("[#a3a3a3]Enter conversation number to resume, or press Enter to cancel[/#a3a3a3]")
    try:
        choice = await ui.line_prompt("resume › ")
    except (EOFError, KeyboardInterrupt):
        return
    choice = choice.strip()
    if not choice:
        return

    # Parse number or id
    selected: dict[str, Any] | None = None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(conversations):
            selected = conversations[idx]
    except ValueError:
        # Try matching by id prefix
        for conv in conversations:
            if conv["id"].startswith(choice):
                selected = conv
                break

    if selected is None:
        ui.console.print(f"[red]no matching conversation for:[/red] {choice}")
        return

    messages = conv_store.load(selected["id"])
    if messages is None:
        ui.console.print(f"[red]failed to load conversation:[/red] {selected['id']}")
        return

    agent.load_messages(messages)
    ui.load_history_from_messages(messages)
    ui.redraw_transcript(str(root_ref["root"]), agent.messages)
    ui.console.print(
        f"[green]•[/] [#9e9e9e]resumed:[/#9e9e9e] "
        f"[bold]{selected['subject']}[/bold] "
        f"([#9e9e9e]{selected['message_count']} msgs, {selected['mtime_label']}[/#9e9e9e])"
    )
    ui.console.print()


async def handle_provider(agent: Agent, ui: TerminalUI, root: Path) -> None:
    """List configured providers and switch the current backend."""
    providers = ProviderConfig.summaries(root)
    ui.print_provider_list(providers)
    ui.console.print("[#a3a3a3]Enter provider number or name, or press Enter to cancel[/#a3a3a3]")
    try:
        choice = (await ui.line_prompt("provider › ")).strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not choice:
        return

    selected = next((provider for provider in providers if provider["name"] == choice), None)
    if selected is None:
        try:
            selected = providers[int(choice) - 1]
        except (ValueError, IndexError):
            selected = None
    if selected is None:
        ui.console.print(f"[red]no matching provider for:[/red] {choice}")
        return

    try:
        backend = load_backend(selected["name"], root)
    except Exception as exc:
        ui.console.print(f"[red]failed to load provider:[/red] {type(exc).__name__}: {exc}")
        return
    agent.set_backend(backend)
    ui.console.print(f"[green]•[/] [#9e9e9e]provider:[/#9e9e9e] [bold]{agent.provider_name}[/bold]")
    ui.console.print()


def configure_runtime(args: argparse.Namespace, root: Path) -> JasmineConfig:
    config = JasmineConfig.load(root)
    if config.model and not args.model:
        os.environ["JASMINE_MODEL"] = config.model
    if config.reasoning_effort and not args.reasoning_effort:
        os.environ["JASMINE_REASONING_EFFORT"] = config.reasoning_effort
    if config.thinking and not args.thinking:
        os.environ["JASMINE_THINKING"] = config.thinking
    if args.fast and args.quality:
        raise SystemExit("Choose either --fast or --quality, not both.")
    if args.fast:
        os.environ["JASMINE_REASONING_EFFORT"] = "high"
        os.environ["JASMINE_THINKING"] = "1"
    if args.quality:
        os.environ["JASMINE_REASONING_EFFORT"] = "xhigh"
        os.environ["JASMINE_THINKING"] = "1"
    if args.model:
        os.environ["JASMINE_MODEL"] = args.model
    if args.reasoning_effort:
        os.environ["JASMINE_REASONING_EFFORT"] = args.reasoning_effort
    if args.thinking:
        os.environ["JASMINE_THINKING"] = "1" if args.thinking == "on" else "0"
    return config


async def async_main() -> None:
    args = parse_args()
    workspace = args.workspace or _load_last_workspace() or "."
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _save_last_workspace(root)

    config = configure_runtime(args, root)

    ui = TerminalUI(root)
    theme = get_theme(config)
    ui.set_theme(theme)
    tools = ToolRegistry(root)
    register_agent_tools(tools)
    backend = load_backend(args.backend, root)
    agent = Agent(backend, tools, ui)
    conv_store = ConversationStore(root)

    # Load command history from the most recent conversation
    conversations = conv_store.list_conversations()
    if conversations:
        latest = conversations[0]
        messages = conv_store.load(latest["id"])
        if messages:
            ui.load_history_from_messages(messages)
    # Start a fresh conversation so new messages don't overwrite the old one
    conv_store.start_conversation()

    root_ref = {"root": root}
    tools_ref = {"tools": tools}

    loop = asyncio.get_running_loop()
    resize_installed = False
    ui.set_redraw_callback(lambda: ui.redraw_transcript(str(root_ref["root"]), agent.messages))
    if hasattr(signal, "SIGWINCH"):
        try:
            loop.add_signal_handler(signal.SIGWINCH, ui.handle_resize)
            resize_installed = True
        except (NotImplementedError, RuntimeError):
            resize_installed = False

    # Clear both the viewport and scrollback so no host shell output remains.
    ui.clear_terminal()
    ui.banner(str(root))
    try:
        while True:
            try:
                raw_text = await ui.prompt()
                user_text = raw_text.strip()
            except (EOFError, KeyboardInterrupt):
                ui.console.print("\nbye")
                return
            if not user_text:
                continue

            try:
                handled = await handle_command(user_text, agent, ui, tools_ref, root_ref, conv_store)
            except EOFError:
                ui.console.print("bye")
                return
            if handled:
                continue

            root = root_ref["root"]
            ui.print_user_message(raw_text)
            compacted, saved_path, original_length = compact_large_paste(root, raw_text, args.paste_threshold)
            if saved_path:
                ui.user_paste_notice(original_length, saved_path)

            task = asyncio.create_task(agent.run_user_turn(compacted))
            signal_installed = False

            def cancel_active_run(signum: int | None = None, frame: Any = None) -> None:
                if not task.done():
                    task.cancel()

            try:
                try:
                    loop.add_signal_handler(signal.SIGINT, cancel_active_run)
                    signal_installed = True
                except (NotImplementedError, RuntimeError):
                    signal_installed = False
                # Keep the handler callable on the UI so prompt_toolkit
                # interactions can re-install it after they corrupt the
                # asyncio-level SIGINT handler (see gh#...).
                ui._interrupt_handler = cancel_active_run
                await task
            except KeyboardInterrupt:
                cancel_active_run()
                with suppress(asyncio.CancelledError):
                    await task
                ui.interrupt_notice()
            except asyncio.CancelledError:
                ui.interrupt_notice()
            except Exception as exc:
                ui.console.print(f"[red]● agent failed:[/red] {type(exc).__name__}: {exc}")
            finally:
                ui._interrupt_handler = None
                if signal_installed:
                    with suppress(Exception):
                        loop.remove_signal_handler(signal.SIGINT)
                ui.reset_phase()
                # Auto-save conversation after each turn
                conv_store.save(agent.messages)
    finally:
        await tools_ref["tools"].close_background_jobs()
        if resize_installed:
            with suppress(Exception):
                loop.remove_signal_handler(signal.SIGWINCH)


def main() -> None:
    if not sys.stdin.isatty():
        asyncio.run(_stdin_mode())
        return
    asyncio.run(async_main())


async def _stdin_mode() -> None:
    args = parse_args()
    workspace = args.workspace or _load_last_workspace() or "."
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _save_last_workspace(root)
    configure_runtime(args, root)
    ui = TerminalUI(root)
    tools = ToolRegistry(root)
    register_agent_tools(tools)
    backend = load_backend(args.backend, root)
    agent = Agent(backend, tools, ui)
    user_text, _saved_path, _original_length = compact_large_paste(root, sys.stdin.read(), args.paste_threshold)
    try:
        await agent.run_user_turn(user_text)
        ui.console.print()
    finally:
        await tools.close_background_jobs()


if __name__ == "__main__":
    main()
