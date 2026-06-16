"""Tests for conversation persistence."""

import tempfile
import unittest
from pathlib import Path

from jasmine.conversation import ConversationStore


class TestConversationStore(unittest.TestCase):
    """Test conversation save, load, and listing."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / ".jasmine").mkdir(exist_ok=True)
        self.store = ConversationStore(self.root)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_start_and_list(self):
        self.store.start_conversation()
        # A fresh conversation has 0 messages, so it won't appear in listing
        # until something is saved. Save a message first.
        self.store.save([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        conversations = self.store.list_conversations()
        self.assertEqual(len(conversations), 1)

    def test_save_and_load(self):
        self.store.start_conversation()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        saved = self.store.save(messages)
        self.assertTrue(saved)

        conversations = self.store.list_conversations()
        self.assertEqual(len(conversations), 1)

        loaded = self.store.load(conversations[0]["id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 3)

    def test_subject_auto_from_first_user(self):
        self.store.start_conversation()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Fix the login bug in auth.py"},
            {"role": "assistant", "content": "ok"},
        ]
        self.store.save(messages)

        conversations = self.store.list_conversations()
        self.assertEqual(conversations[0]["subject"], "Fix the login bug in auth.py")

    def test_empty_conversation_not_saved(self):
        self.store.start_conversation()
        saved = self.store.save([{"role": "system", "content": "sys"}])
        self.assertFalse(saved)

    def test_list_excludes_empty(self):
        # Create and save a conversation
        self.store.start_conversation()
        self.store.save([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
        ])

        # Start another but don't save anything real
        self.store.start_conversation()

        conversations = self.store.list_conversations()
        # Only the first should appear (the empty one has 0 messages)
        self.assertEqual(len(conversations), 1)

    def test_multiple_conversations_newest_first(self):
        import time
        self.store.start_conversation()
        self.store.save([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
        ])
        time.sleep(0.01)  # ensure different mtime

        self.store.start_conversation()
        self.store.save([
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "ok"},
        ])

        conversations = self.store.list_conversations()
        self.assertEqual(len(conversations), 2)
        # Newest first
        subjects = {c["subject"] for c in conversations}
        self.assertIn("first", subjects)
        self.assertIn("second", subjects)
        self.assertEqual(conversations[0]["subject"], "second")

    def test_load_nonexistent(self):
        result = self.store.load("conv_nonexistent")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
