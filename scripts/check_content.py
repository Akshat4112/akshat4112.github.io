#!/usr/bin/env python3
"""Static publishing checks for Hugo Markdown articles."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "content" / "posts"
STATIC = ROOT / "static"
REQUIRED = (
    "title",
    "description",
    "date",
    "lastmod",
    "draft",
    "tags",
    "weight",
    "showtoc",
)


def front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    closing = text.find("\n---\n", 4)
    if closing < 0:
        return {}, text
    raw = text[4:closing]
    values: dict[str, str] = {}
    for line in raw.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values, text[closing + 5 :]


def parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.strip('"').replace("Z", "+00:00"))
    except ValueError:
        return None


def inspect(path: Path) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    meta, body = front_matter(text)
    errors: list[str] = []
    warnings: list[str] = []

    if not meta:
        return ["missing or unclosed YAML front matter"], warnings

    for key in REQUIRED:
        if not meta.get(key):
            errors.append(f"missing required metadata: {key}")

    published = meta.get("draft", "").lower() == "false"
    published_at = parse_date(meta.get("date", ""))
    modified_at = parse_date(meta.get("lastmod", "")) if meta.get("lastmod") else None
    if not published_at:
        errors.append("date is not a valid ISO-8601 timestamp")
    if meta.get("lastmod") and not modified_at:
        errors.append("lastmod is not a valid ISO-8601 timestamp")
    if published_at and modified_at:
        try:
            if modified_at < published_at:
                errors.append("lastmod is earlier than date")
        except TypeError:
            errors.append("date and lastmod use incompatible timezone formats")

    title = meta.get("title", "").strip('"')
    description = meta.get("description", "").strip('"')
    if title and description and title.casefold() in description.casefold():
        warnings.append("description repeats the full title")
    if description and len(description) < 50:
        warnings.append("description is shorter than 50 characters")

    for marker in ("```", "~~~"):
        fence_lines = re.findall(rf"^ {0,3}{re.escape(marker)}(.*)$", body, flags=re.MULTILINE)
        if len(fence_lines) % 2:
            errors.append(f"unbalanced {marker} fenced code block")
        for index, label in enumerate(fence_lines):
            if index % 2 == 0 and not label.strip():
                warnings.append("fenced code block has no language label")

    prose = re.sub(r"^ {0,3}```.*?^ {0,3}```\s*$", "", body, flags=re.MULTILINE | re.DOTALL)
    prose = re.sub(r"^ {0,3}~~~.*?^ {0,3}~~~\s*$", "", prose, flags=re.MULTILINE | re.DOTALL)
    prose = re.sub(r"`[^`\n]+`", "", prose)
    prose = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", "", prose, flags=re.DOTALL)
    prose = re.sub(r"\$\$.*?\$\$|(?<!\$)\$[^$\n]+\$(?!\$)", "", prose, flags=re.DOTALL)
    numeric_citations = re.findall(r"(?<!\!)\[(\d+(?:,\s*\d+)*)\](?!\()", prose)
    if numeric_citations:
        errors.append(
            "uses unexplained numeric citation marker(s): "
            + ", ".join(f"[{marker}]" for marker in sorted(set(numeric_citations)))
        )

    uses_math = bool(re.search(r"(?<!\\)\$\$|(?<!\\)\\\(|(?<!\\)\\\[", body))
    if uses_math and meta.get("math", "").lower() != "true":
        errors.append("contains mathematics but math: true is not set")

    headings = re.findall(r"^(#{2,6})\s+(.+)$", body, flags=re.MULTILINE)
    previous_level = 1
    seen: set[str] = set()
    for marks, heading in headings:
        level = len(marks)
        clean = re.sub(r"[`*_]", "", heading).strip().casefold()
        if level > previous_level + 1:
            warnings.append(f"heading level jumps to H{level}: {heading}")
        if clean in seen:
            warnings.append(f"duplicate heading: {heading}")
        seen.add(clean)
        previous_level = level

    for alt, target in re.findall(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)", body):
        if not alt.strip():
            errors.append(f"image has empty alt text: {target}")
        if target.startswith("/"):
            asset = STATIC / target.lstrip("/")
            if not asset.is_file():
                errors.append(f"local image does not exist: {target}")

    if published and not body.strip():
        errors.append("published article has no body")

    return errors, warnings


def main() -> int:
    article_paths = sorted(p for p in POSTS.glob("*.md") if p.name != "_index.md")
    error_count = 0
    warning_count = 0

    for path in article_paths:
        errors, warnings = inspect(path)
        if errors or warnings:
            print(f"\n{path.relative_to(ROOT)}")
        for message in errors:
            print(f"  ERROR: {message}")
        for message in warnings:
            print(f"  WARN:  {message}")
        error_count += len(errors)
        warning_count += len(warnings)

    print(
        f"\nChecked {len(article_paths)} articles: "
        f"{error_count} error(s), {warning_count} warning(s)."
    )
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
