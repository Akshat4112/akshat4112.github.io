"""Check that the dated Now snapshot renders in both site languages."""

from pathlib import Path

from check_privacy_routes import PageMetadata


def check(path: Path, language: str, title: str, body: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Now route missing: {path}")
    source = path.read_text(encoding="utf-8")
    page = PageMetadata()
    page.feed(source)
    if page.language != language or title not in page.title or body not in source:
        raise SystemExit(f"Incorrect Now page at {path}: lang={page.language!r}, title={page.title!r}")


if __name__ == "__main__":
    output = Path("public")
    check(output / "now/index.html", "en", "Now | Akshat Gupta", "Updated 15 September 2026")
    check(output / "de/now/index.html", "de", "Aktuell | Akshat Gupta", "Aktualisiert am 15. September 2026")
    print("English and German Now pages render independently.")
