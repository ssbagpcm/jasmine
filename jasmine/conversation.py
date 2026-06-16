"""
Conversation persistence: auto-saves agent conversations as JSON files inside
<workspace>/.jasmine/conversations/, exposes a listing for /resume, and supports
loading a previous conversation back into the agent.
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _safe_mtime(conv_path: Path) -> float:
    try:
        return conv_path.stat().st_mtime
    except OSError:
        return 0.0


def _subject_from_first_user(messages: list[dict[str, Any]]) -> str:
    """Derive a conversation subject from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            # Skip compacted paste markers
            if text.startswith("[pasted-content:"):
                text = text.split("\n", 1)[-1].strip() or text
            first_line = text.split("\n", 1)[0].strip()
            return first_line[:120] if len(first_line) > 120 else first_line
    return "Empty conversation"


def _should_update_subject(current_subject: str, latest_user_message: str) -> bool:
    """Return True when the latest user message suggests the topic changed enough
    to justify updating the stored subject."""
    if not latest_user_message.strip():
        return False
    first_line = latest_user_message.strip().split("\n", 1)[0]
    if len(first_line) < 10:
        return False
    # Simple heuristic: if the new first line is very different from the current
    # subject, consider it a topic change.
    overlap = len(set(current_subject.lower().split()) & set(first_line.lower().split()))
    return bool(overlap == 0 and len(first_line) > 30)


class ConversationStore:
    """Manages .jasmine/conversations/ for a given workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.store_dir = self.root / ".jasmine" / "conversations"
        self._index_path = self.store_dir / "_index.json"
        self._current_id: str | None = None
        self._index: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def set_root(self, root: Path) -> None:
        """Switch workspace storage without carrying over the previous id."""
        self.root = root.resolve()
        self.store_dir = self.root / ".jasmine" / "conversations"
        self._index_path = self.store_dir / "_index.json"
        self._current_id = None
        self._index = {}

    def _path_for(self, conv_id: str) -> Path:
        return self.store_dir / f"{conv_id}.json"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _fmt_timestamp(ts: float) -> str:
        dt = datetime.fromtimestamp(ts)
        now = datetime.now()
        if dt.date() == now.date():
            return dt.strftime("%H:%M")
        elif dt.date() == (now - timedelta(days=1)).date():
            return f"yest {dt.strftime('%H:%M')}"
        elif (now.date() - dt.date()).days < 7:
            return dt.strftime("%a %H:%M")
        elif dt.year == now.year:
            return dt.strftime("%b %d %H:%M")
        return dt.strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------
    # index (lightweight metadata cache to avoid reading every file)
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index:
            return self._index
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._index = data
        except Exception:
            self._index = {}
        return self._index

    def _save_index_entry(self, conv_id: str, meta: dict[str, Any]) -> None:
        self._load_index()
        self._index[conv_id] = meta
        try:
            self._ensure_store()
            self._index_path.write_text(
                json.dumps(self._index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def start_conversation(self) -> str:
        """Create a fresh conversation entry and return its id."""
        self._ensure_store()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        conv_id = f"conv_{stamp}"
        record: dict[str, Any] = {
            "id": conv_id,
            "subject": "New conversation",
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "workspace": str(self.root),
            "messages": [],
            "message_count": 0,
        }
        self._path_for(conv_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._current_id = conv_id
        return conv_id

    def save(self, messages: list[dict[str, Any]]) -> bool:
        """Persist the current conversation. Skips empty conversations.
        Returns True if saved, False if skipped (empty)."""
        # Filter out system messages for emptiness check
        user_or_assistant = [
            m for m in messages if m.get("role") in ("user", "assistant")
        ]
        if not user_or_assistant:
            return False  # nothing worth saving

        self._ensure_store()
        if self._current_id is None:
            self.start_conversation()

        assert self._current_id is not None
        path = self._path_for(self._current_id)

        # Load existing record to preserve created_at
        existing: dict[str, Any] = {}
        with suppress(Exception):
            existing = json.loads(path.read_text(encoding="utf-8"))

        subject = existing.get("subject", "New conversation")
        # Auto-set subject from first user message if still default
        if subject == "New conversation":
            subject = _subject_from_first_user(messages)
        else:
            # Check if topic changed
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break
            if _should_update_subject(subject, last_user):
                subject = _subject_from_first_user(messages)

        record: dict[str, Any] = {
            "id": self._current_id,
            "subject": subject,
            "created_at": existing.get("created_at", self._now_iso()),
            "updated_at": self._now_iso(),
            "workspace": str(self.root),
            "messages": messages,
            "message_count": len(messages),
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Update lightweight index so listing doesn't read every file
        self._save_index_entry(self._current_id, {
            "id": record["id"],
            "subject": record["subject"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "workspace": record["workspace"],
            "message_count": record["message_count"],
            "mtime": path.stat().st_mtime,
            "mtime_label": self._fmt_timestamp(path.stat().st_mtime),
        })
        return True

    # ------------------------------------------------------------------
    # read / list
    # ------------------------------------------------------------------

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return metadata for past conversations, newest first.
        Uses a lightweight index file to avoid reading every conversation."""
        self._ensure_store()
        index = self._load_index()
        if index:
            entries = sorted(
                index.values(),
                key=lambda e: e.get("mtime", 0),
                reverse=True,
            )
            return [
                e for e in entries[:limit]
                if e.get("message_count", 0) > 0
            ]

        # Fallback: rebuild index from files (first run / legacy)
        result: list[dict[str, Any]] = []
        for entry in sorted(
            self.store_dir.glob("conv_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
            except Exception:
                continue
            msg_count = data.get("message_count", len(data.get("messages", [])))
            if msg_count == 0:
                continue
            meta = {
                "id": data.get("id", entry.stem),
                "subject": data.get("subject", "Untitled"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "workspace": data.get("workspace", str(self.root)),
                "message_count": msg_count,
                "mtime": _safe_mtime(entry),
                "mtime_label": self._fmt_timestamp(_safe_mtime(entry)),
            }
            result.append(meta)
            self._index[meta["id"]] = meta
            if len(result) >= limit:
                break

        # Persist rebuilt index
        with suppress(Exception):
            self._index_path.write_text(
                json.dumps(self._index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result

    def load(self, conv_id: str) -> list[dict[str, Any]] | None:
        """Load messages from a conversation file. Returns None if not found."""
        path = self._path_for(conv_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        messages = data.get("messages", [])
        if not messages:
            return None
        self._current_id = conv_id
        return messages
