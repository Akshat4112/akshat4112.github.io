"""Check rendered article adjacency, theme controls, and mixed-language archives."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from check_portfolio_refresh import PUBLIC, archive_links


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    children: list[Element | str] = field(default_factory=list)
    language: str = ""

    def has_class(self, name: str) -> bool:
        return name in (self.attrs.get("class") or "").split()

    def descendants(self) -> list[Element]:
        return [node for child in self.children if isinstance(child, Element)
                for node in [child, *child.descendants()]]

    def text(self) -> str:
        return " ".join("".join(child if isinstance(child, str) else child.text()
                                for child in self.children).split())

    def text_languages(self) -> list[tuple[str, str]]:
        return [(child, self.language) for child in self.children
                if isinstance(child, str) and child.strip()] + [
                    fragment for child in self.children if isinstance(child, Element)
                    for fragment in child.text_languages()]


class Document(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.root = Element("document")
        self.stack = [self.root]
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element = Element(tag, values, language=values.get("lang") or self.stack[-1].language)
        self.stack[-1].children.append(element)
        if tag not in self.VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)

    def elements(self, class_name: str) -> list[Element]:
        return [node for node in self.root.descendants() if node.has_class(class_name)]


def output_path(url: str) -> Path:
    return PUBLIC / urlparse(url).path.strip("/") / "index.html"


def main() -> None:
    errors: list[str] = []
    articles = archive_links(PUBLIC / "posts/index.html")
    for path in sorted((PUBLIC / "posts/page").glob("*/index.html"), key=lambda p: int(p.parent.name)):
        articles.extend(archive_links(path))
    if len(articles) < 2 or len(articles) != len(set(articles)):
        raise SystemExit("Expected a nonempty writing archive without repeated articles")

    # The public archive is the reader-visible ordering contract, including all pages.
    for index, url in enumerate(articles):
        document = Document(output_path(url))
        navigation = document.elements("paginav")
        if len(navigation) != 1:
            errors.append(f"{url}: expected one article navigation")
            continue
        for direction, expected in (("prev", articles[index - 1] if index else None),
                                    ("next", articles[index + 1] if index + 1 < len(articles) else None)):
            links = [node for node in navigation[0].descendants()
                     if node.tag == "a" and node.has_class(direction)]
            actual = [urlparse(node.attrs.get("href") or "").path for node in links]
            if actual != ([expected] if expected else []):
                errors.append(f"{url}: {direction} is {actual}, expected {expected}")

    labels = {"en": "Toggle colour theme", "de": "Farbschema wechseln"}
    languages: set[str] = set()
    article_paths = {output_path(url) for url in articles}
    for path in sorted(PUBLIC.rglob("*.html")):
        document = Document(path)
        if path not in article_paths and document.elements("paginav"):
            errors.append(f"{path.relative_to(PUBLIC)}: article navigation on a non-article page")
        buttons = [node for node in document.root.descendants()
                   if node.tag == "button" and node.attrs.get("id") == "theme-toggle"]
        # Hugo also emits redirect documents without the shared page header.
        if not any(node.tag == "header" and node.has_class("header")
                   for node in document.root.descendants()):
            continue
        if len(buttons) != 1:
            errors.append(f"{path.relative_to(PUBLIC)}: expected one theme button")
            continue
        button = buttons[0]
        language = button.language.split("-")[0]
        languages.add(language)
        if button.attrs.get("aria-label") != labels.get(language):
            errors.append(f"{path.relative_to(PUBLIC)}: theme button lacks the {language} accessible name")
        if button.attrs.get("accesskey") != "t" or "Alt + T" not in (button.attrs.get("title") or ""):
            errors.append(f"{path.relative_to(PUBLIC)}: theme shortcut missing")
        icons = [node for node in button.descendants() if node.tag == "svg"]
        if len(icons) != 2 or any(node.attrs.get("aria-hidden") != "true"
                                  or node.attrs.get("focusable") != "false" for node in icons):
            errors.append(f"{path.relative_to(PUBLIC)}: theme icons must be decorative")
    if languages != set(labels):
        errors.append(f"Expected both language variants, found {languages}")

    mixed_entries = 0
    for section in ("posts", "publications"):
        archive = Document(PUBLIC / "de" / section / "index.html")
        entries = archive.elements("post-entry")
        if not entries:
            errors.append(f"German {section} archive is empty")
        for entry in entries:
            mixed_entries += 1
            nodes = entry.descendants()
            link = next(node for node in nodes if node.tag == "a" and node.has_class("entry-link"))
            source = Document(output_path(link.attrs.get("href") or ""))
            heading = next(node for node in nodes if node.tag == "h2")
            badge = next(node for node in nodes if node.has_class("language-badge"))
            title = source.elements("post-title")[0].text()
            fragments = [(text.strip(), language) for text, language in heading.text_languages()
                         if text.strip() != badge.text()]
            if " ".join(text for text, _ in fragments) != title or any(
                    language.split("-")[0] != "en" for _, language in fragments):
                errors.append(f"{section}: English title must match its article and have lang=en: {title}")
            summaries = [node for node in nodes if node.has_class("entry-content")]
            descriptions = source.elements("post-description")
            if len(summaries) != len(descriptions):
                errors.append(f"{section}: archive description missing or unexpected: {title}")
            for summary, description in zip(summaries, descriptions):
                if summary.text() != description.text() or any(
                        language.split("-")[0] != "en" for _, language in summary.text_languages()):
                    errors.append(f"{section}: English description must match its article and have lang=en: {title}")
            for german in [badge, *[node for node in nodes if node.has_class("entry-footer")]]:
                if not german.text() or any(language.split("-")[0] != "de"
                                           for _, language in german.text_languages()):
                    errors.append(f"{section}: German badge and metadata must retain German language: {title}")
            if badge.text() != "Englisch":
                errors.append(f"{section}: language badge must remain translated: {title}")

    if errors:
        raise SystemExit("Navigation/accessibility checks failed:\n" + "\n".join(errors))
    print(f"Article adjacency and endpoints verified across {len(articles)} articles; "
          f"bilingual theme controls and {mixed_entries} mixed-language archive entries verified.")


if __name__ == "__main__":
    main()
