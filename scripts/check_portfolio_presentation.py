"""Check the rendered bilingual landing-page entry points."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a":
            self.links.append(values.get("href") or "")
        if values.get("id"):
            self.ids.add(str(values["id"]))


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "public"
    for language, about in (("", "/about/"), ("de", "/de/about/")):
        parser = Links()
        parser.feed((root / language / "index.html").read_text())
        assert "#work" in parser.links, f"{language or 'en'}: missing direct project CTA"
        assert "work" in parser.ids, "Project CTA must have a destination"
        assert about in parser.links, "Background must remain discoverable"
    for path in root.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text())
        assert not any("1Qj6kaZXM40ixUgOAewhZlTj57piWERSw" in link for link in parser.links), f"Outdated CV remains linked: {path}"
    print("Bilingual project entry points and removal of outdated CV links verified.")


if __name__ == "__main__":
    main()
