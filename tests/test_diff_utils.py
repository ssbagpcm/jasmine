"""Tests for diff_utils: patch application and generation."""

import unittest

from jasmine.diff_utils import (
    apply_codex_update_to_text,
    apply_unified_patch_to_text,
    make_unified_diff,
    split_multi_file_diff,
)


class TestUnifiedPatch(unittest.TestCase):
    """Test applying unified diff patches."""

    def test_simple_replacement(self):
        original = "line1\nline2\nline3\n"
        patch = "@@ -1,3 +1,3 @@\n line1\n-line2\n+modified line2\n line3\n"
        result = apply_unified_patch_to_text(original, patch)
        self.assertEqual(result, "line1\nmodified line2\nline3\n")

    def test_add_line(self):
        original = "line1\nline2\n"
        patch = "@@ -1,2 +1,3 @@\n line1\n+new line\n line2\n"
        result = apply_unified_patch_to_text(original, patch)
        self.assertEqual(result, "line1\nnew line\nline2\n")

    def test_delete_line(self):
        original = "line1\nline2\nline3\n"
        patch = "@@ -1,3 +1,2 @@\n line1\n-line2\n line3\n"
        result = apply_unified_patch_to_text(original, patch)
        self.assertEqual(result, "line1\nline3\n")

    def test_multiple_hunks(self):
        original = "a\nb\nc\nd\ne\nf\n"
        patch = (
            "@@ -1,3 +1,3 @@\n a\n-b\n+BB\n c\n"
            "@@ -4,3 +4,3 @@\n d\n-e\n+EE\n f\n"
        )
        result = apply_unified_patch_to_text(original, patch)
        self.assertEqual(result, "a\nBB\nc\nd\nEE\nf\n")

    def test_context_mismatch_raises(self):
        original = "line1\nline2\n"
        patch = "@@ -1,2 +1,2 @@\n line1\n-wrong context\n"
        with self.assertRaises(ValueError):
            apply_unified_patch_to_text(original, patch)

    def test_trailing_whitespace_tolerance(self):
        original = "line1   \nline2\n"
        patch = "@@ -1,2 +1,2 @@\n line1\n-line2\n+line3\n"
        result = apply_unified_patch_to_text(original, patch)
        self.assertEqual(result, "line1   \nline3\n")


class TestCodexPatch(unittest.TestCase):
    """Test Codex-style update hunks."""

    def test_simple_update(self):
        before = "function hello() {\n    console.log('old');\n}\n"
        section = [
            " function hello() {",
            "-    console.log('old');",
            "+    console.log('new');",
            " }",
        ]
        result = apply_codex_update_to_text(before, section)
        self.assertIn("console.log('new')", result)
        self.assertNotIn("console.log('old')", result)

    def test_ambiguous_context_raises(self):
        # Context that appears multiple times in the source without unique removal lines
        before = "line\napple\nline\nbanana\n"
        section = [
            " line",
            "-apple",
            "+cherry",
        ]
        # "line\napple\n" appears once at pos 0, so this is unique. Not ambiguous.
        # We need a case where old appears 0 or 2+ times. Let's use context with no
        # removal lines so old = context only, which may repeat:
        before = "dup\na\ndup\nb\n"
        section = [
            " dup",
            "+new",
        ]
        # old = "dup\n", which appears twice (positions 0 and 6). Should raise.
        with self.assertRaises(ValueError):
            apply_codex_update_to_text(before, section)


class TestMakeUnifiedDiff(unittest.TestCase):
    """Test unified diff generation."""

    def test_generates_valid_diff(self):
        before = "line1\nline2\nline3\n"
        after = "line1\nmodified\nline3\n"
        diff = make_unified_diff("test.txt", before, after)
        self.assertIn("--- a/test.txt", diff)
        self.assertIn("+++ b/test.txt", diff)
        self.assertIn("-line2", diff)
        self.assertIn("+modified", diff)

    def test_no_change_empty_diff(self):
        text = "unchanged\n"
        diff = make_unified_diff("f.txt", text, text)
        # No changes means empty diff string
        self.assertEqual(diff, "")


class TestSplitMultiFileDiff(unittest.TestCase):
    """Test splitting multi-file diffs."""

    def test_single_file(self):
        diff = "diff --git a/x.txt b/x.txt\n--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
        chunks = split_multi_file_diff(diff)
        self.assertEqual(len(chunks), 1)
        self.assertIn("x.txt", chunks)

    def test_multiple_files(self):
        diff = (
            "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
            "diff --git a/b.txt b/b.txt\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-old2\n+new2\n"
        )
        chunks = split_multi_file_diff(diff)
        self.assertEqual(len(chunks), 2)
        self.assertIn("a.txt", chunks)
        self.assertIn("b.txt", chunks)


if __name__ == "__main__":
    unittest.main()
