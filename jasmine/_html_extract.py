"""Structured HTML content extraction for web_extract tool."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin


def extract_structured(html: str, url: str, max_chars: int) -> dict[str, Any]:
    """Extract a rich structured representation from HTML."""
    from bs4 import BeautifulSoup

    result: dict[str, Any] = {
        "title": "",
        "description": "",
        "text": "",
        "headings": [],
        "links": [],
        "images": [],
        "meta": {},
        "structured_data": [],
    }
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return result

    # --- Title ---
    title_tag = soup.find("title")
    if title_tag:
        result["title"] = title_tag.get_text(strip=True)
    if not result["title"]:
        og_title = soup.find("meta", property="og:title")
        if og_title:
            result["title"] = og_title.get("content", "")

    # --- Meta description ---
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or "").lower()
        prop = (meta.get("property") or "").lower()
        content = meta.get("content", "")
        if name in ("description", "keywords", "author", "robots") and content:
            result["meta"][name] = content
        if prop in ("og:title", "og:description", "og:image", "og:url", "og:type") and content:
            result["meta"][prop] = content
    result["description"] = result["meta"].get("description") or result["meta"].get("og:description", "")

    # --- Headings ---
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            text = tag.get_text(strip=True)
            if text and len(text) > 1:
                result["headings"].append({"level": level, "text": text})

    # --- Links ---
    seen_urls: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href = urljoin(url, a_tag["href"])
        text = a_tag.get_text(strip=True)
        if href and href not in seen_urls and not href.startswith("javascript:"):
            seen_urls.add(href)
            result["links"].append({"text": text or "", "href": href})

    # --- Images ---
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if src:
            src = urljoin(url, src)
        alt = img.get("alt") or ""
        if src:
            result["images"].append({"src": src, "alt": alt})

    # --- Structured data (JSON-LD) ---
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                result["structured_data"].extend(data)
            elif isinstance(data, dict):
                result["structured_data"].append(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- SSR data (__NEXT_DATA__, __NUXT__, etc.) ---
    ssr_patterns = [
        r'(?:window\.)?__NEXT_DATA__\s*=\s*({[\s\S]*?});',
        r'(?:window\.)?__NUXT__\s*=\s*({[\s\S]*?});',
        r'(?:window\.)?__INITIAL_STATE__\s*=\s*({[\s\S]*?});',
    ]
    for script in soup.find_all("script"):
        content = script.string
        if not content:
            continue
        for pattern in ssr_patterns:
            for match in re.finditer(pattern, content):
                try:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict):
                        result["structured_data"].append({"source": "ssr_payload", "data": parsed})
                except (json.JSONDecodeError, ValueError):
                    pass

    # --- Text content ---
    try:
        import html2text
        from readability import Document

        doc = Document(html)
        title_r = doc.title()
        if title_r and not result["title"]:
            result["title"] = title_r
        summary_html = doc.summary()
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        text = h.handle(summary_html).strip()
        if len(text) < 100:
            h.ignore_links = True
            text = h.handle(html).strip()
    except Exception:
        text = soup.get_text(separator="\n", strip=True)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated: {len(text) - max_chars} more chars]"
    result["text"] = text
    result["text_length"] = len(text)

    return result
