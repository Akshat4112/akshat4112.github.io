"""Fail the Hugo build if the bilingual privacy notices collide or disappear."""

from html.parser import HTMLParser
from pathlib import Path


class PageMetadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.language = None
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "html":
            self.language = dict(attrs).get("lang")
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def check(path: Path, language: str, title: str, text: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Privacy route missing: {path}")
    source = path.read_text(encoding="utf-8")
    metadata = PageMetadata()
    metadata.feed(source)
    if metadata.language != language or title not in metadata.title or text not in source:
        raise SystemExit(
            f"Incorrect privacy page at {path}: lang={metadata.language!r}, "
            f"title={metadata.title!r}"
        )


if __name__ == "__main__":
    output = Path("public")
    check(output / "privacy/index.html", "en", "Privacy and analytics", "This personal portfolio")
    check(output / "de/privacy/index.html", "de", "Datenschutz und Analyse", "Dieses persönliche Portfolio")
    print("English and German privacy routes render independently.")
