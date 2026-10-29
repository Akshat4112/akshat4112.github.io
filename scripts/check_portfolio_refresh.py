"""Check project discovery and chronological archives in the built portfolio."""

from __future__ import annotations

import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
PROJECTS = ("pdf-tools", "immopilot-de", "nettopilot-de")


class ArchiveLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and "entry-link" in (values.get("class") or "").split():
            self.links.append(urlparse(values.get("href") or "").path)


def archive_links(path: Path) -> list[str]:
    page = ArchiveLinks()
    page.feed(path.read_text(encoding="utf-8"))
    return page.links


def main() -> None:
    for language in ("", "de/"):
        homepage = (PUBLIC / language / "index.html").read_text(encoding="utf-8")
        for project in PROJECTS:
            if f"https://github.com/Akshat4112/{project}" not in homepage:
                raise SystemExit(f"{language or 'English'} homepage omits {project}")
        if "/posts/pdf-tools-output-validation/" not in homepage:
            raise SystemExit(f"{language or 'English'} homepage omits the engineering case study")

    english = archive_links(PUBLIC / "posts/index.html")
    for path in sorted((PUBLIC / "posts/page").glob("*/index.html"), key=lambda p: int(p.parent.name)):
        english.extend(archive_links(path))
    german = archive_links(PUBLIC / "de/posts/index.html")
    if not english or len(set(english)) != len(english) or english != german:
        raise SystemExit("Writing archives must contain the same articles in the same order, without duplicates")

    dates: list[datetime] = []
    for link in english:
        source = (PUBLIC / link.strip("/") / "index.html").read_text(encoding="utf-8")
        objects = [json.loads(block) for block in re.findall(
            r'<script\s+type=["\']?application/ld\+json["\']?>(.*?)</script>', source, re.DOTALL
        )]
        post = next(item for item in objects if item.get("@type") == "BlogPosting")
        dates.append(datetime.fromisoformat(post["datePublished"].replace("Z", "+00:00")))
    if dates != sorted(dates, reverse=True):
        raise SystemExit("Writing archives are not ordered newest first")
    if "/posts/pdf-tools-output-validation/" not in english:
        raise SystemExit("The engineering case study is missing from the writing archives")
    print(f"Project discovery and bilingual chronology verified across {len(english)} articles.")


if __name__ == "__main__":
    main()
