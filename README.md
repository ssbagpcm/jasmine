# Jasmine

Jasmine is a compact terminal coding agent with streaming Markdown, command
approval, precise patches, live plans, background sessions, and API usage
tracking.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
jasmine .
```

Offline smoke test:

```bash
jasmine --backend mock .
```

## Model-Facing Tools

Jasmine exposes a compact tool surface:

- `exec_command`: inspect code with `rg`, `sed`, and Git; run focused checks.
- `write_stdin`: interact with a background shell or PTY session.
- `terminal_screen`: inspect a model-controlled PTY screen and cursor.
- `terminal_close`: close an interactive PTY and its child processes.
- `apply_patch`: edit files with simple exact replacements or advanced patches.
- `view_image`: inspect local images and attach a thumbnail for vision-capable providers.
- `web_search`: search the web for current information or docs.
- `web_extract`: extract readable text and structured data from a web page.
- `list_skills`: discover reusable workspace and built-in workflows.
- `read_skill`: load one relevant workflow without scanning unrelated guidance.
- `ask_user`: ask for a missing decision instead of guessing.
- `multi_tool_use_parallel`: batch 1-4 independent read-only inspections, including web and image reads.
- `update_plan`: keep one short visible implementation plan.

Codebase search uses focused `rg` commands through `exec_command`. Jasmine does
not maintain a noisy pseudo-database or expose overlapping search wrappers.

## Skills

Jasmine ships with reusable skills under `jasmine/skills/`. A workspace can add
or override skills with `skills/<name>/SKILL.md` or
`.jasmine/skills/<name>/SKILL.md`. The model lists skills only when specialized
work may benefit from one, then reads the matching skill by name.

## Context And Tokens

- Tool calls use the provider-native assistant/tool protocol with
  `tool_call_id`.
- Repeated read-only shell commands are reused within a turn.
- Identical tool calls are stopped after two attempts.
- Shell output enters model context unchanged. A complete `cat file` remains a
  complete file read in conversation history.
- Conversation history is preserved normally without automatic compaction.
- `/usage` shows API request, prompt, completion, reasoning, and cache-hit token
  totals.
- Large pasted prompts are preserved by default. `--paste-threshold` can opt in
  to saving large pastes under `.jasmine/pastes/`.
- Large user messages keep their full model context while the terminal shows a
  short `[Pasted Content N chars]` preview.

## Commands

- `/menu`: show the menu.
- `/tools`: show the model-facing tools.
- `/provider`: switch provider from the configured-provider menu.
- `/resume`: load a saved conversation and replay its visible transcript.
- `/restart`: clear and rebuild the UI without losing the current conversation.
- `/usage`: show API token usage.
- `/trusted`: show remembered shell prefixes.
- `/clear`: clear chat context.
- `/cd <path>`: switch workspace.
- `/exit`: quit.

## Runtime Modes

- Default DeepSeek mode: thinking enabled with `xhigh` reasoning effort.
- `--fast`: keep DeepSeek thinking enabled with `high` reasoning effort.
- `--quality`: enable DeepSeek thinking mode with `xhigh` reasoning effort.
- `--model`, `--reasoning-effort`, and `--thinking`: explicit overrides.

## Providers

OpenAI-compatible providers live in `.jasmine/providers/<name>.toml` for a
workspace or `~/.jasmine/providers/<name>.toml` for all workspaces. A workspace
file takes precedence. Select one with `--backend <name>` or switch at runtime
with `/provider`.

```toml
base_url = "https://api.deepseek.com"
model = "deepseek-v4-pro"
api_key = "..."
supports_thinking = true
thinking_default = true
reasoning_effort = "max"
input_price_per_million = 0.435
cached_input_price_per_million = 0.003625
output_price_per_million = 0.87
```

Pricing fields are optional. When present, `/usage` includes the estimated API
cost.

## Architecture

```
jasmine/
├── main.py           # CLI entry point, argument parsing, command dispatch
├── agent.py          # Model loop: streaming, tool dispatch, nudge logic
├── ai.py             # AIBackend protocol + MockBackend + loader
├── openai_backend.py # Generic OpenAI-compatible streaming backend
├── deepseek_backend.py # DeepSeek-specific defaults (extends OpenAIBackend)
├── providers.py      # Multi-provider TOML config loading
├── tools.py          # ToolRegistry + all built-in tool implementations
├── ui.py             # TerminalUI: Rich rendering, prompt_toolkit input
├── terminal.py       # PTY management, command capture, keyboard encoding
├── config.py         # .jasmine.toml workspace config + theme definitions
├── conversation.py   # Auto-save/load/resume of agent conversations
├── diff_utils.py     # Unified diff application and generation
├── prompts.py        # System prompt (Ultra Mode instruction set)
└── skills/           # Built-in reusable skill workflows
```

## Code Style

- 120-char lines, `from __future__ import annotations` in every file, type hints on public methods.

### System prompt
The system prompt lives in `jasmine/prompts.py`. It is a single continuous
Markdown string sent as the first message of every conversation. Providers that
support prompt caching (DeepSeek, Anthropic) will automatically cache repeated
prefix content — keep structural changes minimal and prefer appending
instructions at the end when possible.
