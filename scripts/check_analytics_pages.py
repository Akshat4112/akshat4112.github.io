"""Verify only rendered article pages opt in to reading-depth analytics."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class PageMarker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.is_article = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("name") == "portfolio-content-type":
            self.is_article = values.get("content") == "post"


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "public"
    posts = [path for path in (root / "posts").glob("*/index.html") if path.parent.name != "page"]
    assert posts, "No articles found"
    for path in posts:
        page = PageMarker()
        page.feed(path.read_text())
        assert page.is_article, f"Missing article marker: {path}"
    for route in ("", "de", "about", "de/about", "now", "de/now", "privacy", "de/privacy", "posts", "de/posts", "publications/glyphnet"):
        path = root / route / "index.html"
        page = PageMarker()
        page.feed(path.read_text())
        assert not page.is_article, f"Ordinary page marked as article: {route}"
    print(f"Reading analytics enabled only for {len(posts)} articles; ordinary pages excluded.")


if __name__ == "__main__":
    main()
