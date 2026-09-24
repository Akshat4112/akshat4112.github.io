#!/usr/bin/env python3
"""Catch Markdown/HTML parsing that truncates equations and article word counts."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path


PUBLIC_POSTS = Path(__file__).resolve().parents[1] / "public" / "posts"
MIN_WORD_COUNTS = {
    "model-extraction-attacks": 1800,
    "llm-fine-tuning-lora": 1700,
    "memories-in-large-language-models": 1600,
}
EXPECTED_MATH = {
    "model-extraction-attacks": (r"x_{\lt i}, y_{\lt i}, K_A",),
    "llm-fine-tuning-lora": (r"y_{\lt t}",),
    "memories-in-large-language-models": (r"x_{\lt t}",),
    "speaker-anonymization": (r"q\lt\tau",),
}


class ArticleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def inspect(path: Path) -> list[str]:
    html = path.read_text(encoding="utf-8")
    errors: list[str] = []
    article = re.search(
        r'<div class=["\']?post-content["\']?>(.*?)<footer class=["\']?post-footer',
        html,
        re.DOTALL,
    )
    if not article:
        return [f"{path}: post content not found"]

    content = article.group(1)
    for block in re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL):
        if re.search(r"</?(?:p|ul|li)\b", block):
            errors.append(f"{path}: Markdown split a display equation into paragraphs or a list")

    parsed = ArticleText()
    parsed.feed(content)
    visible_text = "".join(parsed.parts)
    for expression in EXPECTED_MATH.get(path.parent.name, ()):
        if expression not in visible_text:
            errors.append(f"{path}: equation text missing from the parsed article: {expression}")

    min_words = MIN_WORD_COUNTS.get(path.parent.name)
    if min_words:
        scripts = re.findall(
            r"<script\s+type=[\"']?application/ld\+json[\"']?>(.*?)</script>",
            html,
            re.DOTALL,
        )
        posts = [data for script in scripts if (data := json.loads(script)).get("@type") == "BlogPosting"]
        if len(posts) != 1 or int(posts[0].get("wordCount", 0)) < min_words:
            errors.append(f"{path}: BlogPosting wordCount is missing or below {min_words}")

    return errors


def main() -> None:
    paths = sorted(PUBLIC_POSTS.glob("*/index.html"))
    if not paths:
        raise SystemExit("No built posts found; run hugo --minify first")
    errors = [error for path in paths for error in inspect(path)]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked display math and article text in {len(paths)} built posts")


if __name__ == "__main__":
    main()
