"""Integration tests: agent loop with MockBackend and tool dispatch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jasmine.agent import Agent
from jasmine.ai import MockBackend
from jasmine.main import register_agent_tools
from jasmine.tools import ToolRegistry
from jasmine.ui import TerminalUI


class TestAgentIntegration(unittest.IsolatedAsyncioTestCase):
    """Full agent loop with MockBackend."""

    async def asyncSetUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / ".jasmine").mkdir(exist_ok=True)
        self.ui = TerminalUI(self.root)
        self.tools = ToolRegistry(self.root)
        register_agent_tools(self.tools)
        self.backend = MockBackend()
        self.agent = Agent(self.backend, self.tools, self.ui)

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_mock_backend_text_response(self):
        """A plain message gets a text response without tool calls."""
        await self.agent.run_user_turn("hello")
        # Should have system + user + assistant (text) in messages
        roles = [m["role"] for m in self.agent.messages]
        self.assertIn("system", roles)
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    async def test_read_command_triggers_tool_call(self):
        """MockBackend /read triggers an exec_command tool call."""
        test_file = self.root / "test.py"
        test_file.write_text("print('hello')\n")

        await self.agent.run_user_turn(f"/read {test_file}")

        tool_messages = [
            m for m in self.agent.messages
            if m.get("role") == "tool"
        ]
        self.assertGreaterEqual(len(tool_messages), 1,
                                "Expected at least one tool response")

    async def test_run_command_triggers_exec(self):
        """MockBackend /run triggers exec_command."""
        await self.agent.run_user_turn("/run echo ok")

        tool_messages = [
            m for m in self.agent.messages
            if m.get("role") == "tool"
        ]
        self.assertGreaterEqual(len(tool_messages), 1)

    async def test_agent_tracks_usage(self):
        """Agent increments request counter after a turn."""
        before = self.agent.usage["requests"]
        await self.agent.run_user_turn("test")
        self.assertGreater(self.agent.usage["requests"], before)

    async def test_clear_context_resets_messages(self):
        """clear_context resets to system message only."""
        await self.agent.run_user_turn("hello")
        self.agent.clear_context()
        self.assertEqual(len(self.agent.messages), 1)
        self.assertEqual(self.agent.messages[0]["role"], "system")

    async def test_load_messages_replaces_conversation(self):
        """load_messages replaces the current conversation."""
        saved = [
            {"role": "user", "content": "saved message"},
            {"role": "assistant", "content": "saved response"},
        ]
        self.agent.load_messages(saved)
        # Should have system + user + assistant
        roles = [m["role"] for m in self.agent.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(self.agent.messages[1]["content"], "saved message")

    async def test_tool_registry_has_all_expected_tools(self):
        """All expected tools are registered after register_agent_tools."""
        names = set(self.tools.tools.keys())
        expected = {
            "exec_command", "write_stdin", "terminal_screen",
            "terminal_close", "apply_patch", "view_image",
            "list_skills", "read_skill",
            "multi_tool_use_parallel", "update_plan", "ask_user",
            "web_search", "web_extract",
        }
        for name in expected:
            self.assertIn(name, names, f"Missing tool: {name}")

    async def test_plan_update_via_agent(self):
        """update_plan tool works through the agent's tool call path."""
        result = self.agent._update_plan({
            "plan": [
                {"step": "Test step", "status": "in_progress"},
            ],
        })
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["plan"]), 1)
        self.assertEqual(result["plan"][0]["text"], "Test step")

    async def test_ask_user_rejects_empty_question(self):
        """ask_user returns error for empty question."""
        result = await self.agent._ask_user({"question": ""})
        self.assertFalse(result["ok"])
        self.assertIn("question is required", result["error"])

    async def test_normalize_tool_aliases(self):
        """Agent normalizes tool name aliases correctly."""
        name, args = self.agent._normalize_tool_call("shell", {"cmd": "ls"})
        self.assertEqual(name, "exec_command")
        self.assertIn("command", args)

        name, args = self.agent._normalize_tool_call("bash", {"command": "ls"})
        self.assertEqual(name, "exec_command")

        name, args = self.agent._normalize_tool_call("stdin", {"input": "text"})
        self.assertEqual(name, "write_stdin")
        self.assertIn("chars", args)

    async def test_validate_tool_call_catches_unbounded_cat(self):
        """Agent warns about unbounded cat without arguments."""
        invalid = self.agent._validate_tool_call("exec_command", {"command": "cat"})
        self.assertIsNotNone(invalid)
        self.assertFalse(invalid["ok"])

    async def test_validate_tool_cat_with_file_is_ok(self):
        """Agent allows cat with a target file."""
        invalid = self.agent._validate_tool_call(
            "exec_command", {"command": "cat file.txt"}
        )
        self.assertIsNone(invalid)

    async def test_missing_command_remembered(self):
        """Agent remembers missing shell commands to avoid retries."""
        self.agent._remember_missing_shell_command(
            "exec_command",
            {"command": "nonexistent_cmd --help"},
            {"returncode": 127, "output": "command not found: nonexistent_cmd"},
        )
        invalid = self.agent._validate_tool_call(
            "exec_command", {"command": "nonexistent_cmd --help"}
        )
        self.assertIsNotNone(invalid)
        self.assertIn("unavailable", invalid["error"])


if __name__ == "__main__":
    unittest.main()
