from __future__ import annotations

import difflib
import re


def make_unified_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="\n",
        )
    )


def _normalize_line(line: str) -> str:
    return line.rstrip(" \t\r\n")


def _lines_match(source_line: str, patch_line: str, tolerant: bool) -> bool:
    if not tolerant:
        return source_line == patch_line
    return _normalize_line(source_line) == _normalize_line(patch_line)


def _raise_context_error(
    message: str,
    src_index: int,
    source: list[str],
    expected: str,
    hunk_num: int,
) -> None:
    start = max(0, src_index - 3)
    end = min(len(source), src_index + 3)
    context = "\n".join(
        f"{'>>>' if index == src_index else '   '} {index + 1}: {source[index].rstrip()}"
        for index in range(start, end)
    )
    got = source[src_index].rstrip() if src_index < len(source) else "<end of file>"
    raise ValueError(
        f"{message} at hunk {hunk_num}, line {src_index + 1}\n"
        f"  Expected: {expected.rstrip()!r}\n"
        f"  Got:      {got!r}\n"
        f"  Surrounding context:\n{context}"
    )


def apply_unified_patch_to_text(original: str, patch: str) -> str:
    """Apply a unified diff conservatively.

    The second pass tolerates trailing whitespace differences only. It never
    slides hunks or accepts substring matches, because those can edit the wrong
    location.
    """
    for tolerant in (False, True):
        try:
            return _apply_unified_patch_inner(original, patch, tolerant=tolerant)
        except ValueError:
            if tolerant:
                raise
    return original


def _apply_unified_patch_inner(original: str, patch: str, tolerant: bool) -> str:
    source = original.splitlines(keepends=True)
    result: list[str] = []
    src_index = 0
    patch_lines = patch.splitlines(keepends=True)

    while patch_lines and not patch_lines[0].startswith("@@"):
        patch_lines.pop(0)

    hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    index = 0
    hunk_num = 0
    while index < len(patch_lines):
        match = hunk_header.match(patch_lines[index])
        if not match:
            raise ValueError(f"Invalid unified diff hunk header: {patch_lines[index].rstrip()}")
        target_index = int(match.group(1)) - 1
        hunk_num += 1
        if target_index < src_index:
            raise ValueError("Overlapping or out-of-order patch hunks")
        result.extend(source[src_index:target_index])
        src_index = target_index
        index += 1

        while index < len(patch_lines) and not patch_lines[index].startswith("@@"):
            line = patch_lines[index]
            index += 1
            if line.startswith("\\ No newline at end of file") or not line:
                continue
            marker, payload = line[0], line[1:]
            if marker == "+":
                result.append(payload)
                continue
            if marker not in {" ", "-"}:
                raise ValueError(f"Invalid patch line marker {marker!r} at hunk {hunk_num}")
            if src_index >= len(source) or not _lines_match(source[src_index], payload, tolerant):
                _raise_context_error("Patch context mismatch", src_index, source, payload, hunk_num)
            if marker == " ":
                result.append(source[src_index])
            src_index += 1

    result.extend(source[src_index:])
    return "".join(result)


def split_multi_file_diff(diff_text: str) -> dict[str, str]:
    lines = diff_text.splitlines(keepends=True)
    chunks: dict[str, list[str]] = {}
    current_path: str | None = None
    current: list[str] = []

    def flush() -> None:
        nonlocal current_path, current
        if current_path and current:
            chunks.setdefault(current_path, []).extend(current)
        current_path = None
        current = []

    for line in lines:
        if line.startswith("diff --git "):
            flush()
        if line.startswith("--- ") and current and any(item.startswith("@@") for item in current):
            flush()
        if line.startswith("+++ "):
            raw = line[4:].strip()
            current_path = raw[2:] if raw.startswith("b/") else raw
            if current_path == "/dev/null":
                current_path = None
        current.append(line)
    flush()
    return {path: "".join(chunk) for path, chunk in chunks.items()}


def apply_codex_update_to_text(before: str, section: list[str]) -> str:
    """Apply Codex-style update hunks using exact, unique context."""
    text = before
    hunks: list[list[str]] = []
    current: list[str] = []
    for line in section:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        hunks.append(current)

    for hunk in hunks:
        old_lines: list[str] = []
        new_lines: list[str] = []
        for raw in hunk:
            if not raw:
                old_lines.append("\n")
                new_lines.append("\n")
                continue
            marker, payload = raw[0], raw[1:] + "\n"
            if marker == " ":
                old_lines.append(payload)
                new_lines.append(payload)
            elif marker == "-":
                old_lines.append(payload)
            elif marker == "+":
                new_lines.append(payload)
            else:
                raise ValueError(f"Invalid update line marker {marker!r}")

        old = "".join(old_lines)
        new = "".join(new_lines)
        if not old:
            raise ValueError("Update hunks need unchanged or removed context")

        # Use str.find in a loop — much faster than re.finditer(re.escape(...))
        # for large files.
        def _find_all(haystack: str, needle: str) -> list[int]:
            positions: list[int] = []
            start = 0
            while True:
                pos = haystack.find(needle, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            return positions

        positions = _find_all(text, old)
        if not positions and old.endswith("\n"):
            trimmed = old[:-1]
            positions = _find_all(text, trimmed)
            if positions:
                old = trimmed
                new = new[:-1] if new.endswith("\n") else new
        if len(positions) != 1:
            reason = "not found" if not positions else "ambiguous"
            raise ValueError(f"Patch context {reason}; include more unchanged lines")
        start = positions[0]
        text = text[:start] + new + text[start + len(old) :]
    return text
