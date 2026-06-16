"""Compact Markdown rendering for Jasmine terminal output."""

from __future__ import annotations

from typing import Any

from rich.console import Console
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
from rich.text import Text
from rich.theme import Theme as RichTheme

SUCCESS_STYLE = "bold #22c55e"

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
