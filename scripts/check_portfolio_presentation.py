"""Check the rendered bilingual landing-page entry points."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from check_navigation_accessibility import Document


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
    for prefix, experience in (("", "experience"), ("/de", "berufserfahrung")):
        local = root / prefix.strip("/")
        homepage = Document(local / "index.html")
        portfolio = homepage.elements("portfolio-home")[0]
        headings = [node.attrs.get("id") for node in portfolio.descendants()
                    if node.tag in {"h1", "h2"}]
        assert headings == ["intro-title", "work-title", "writing-title", "about-title"], headings
        intro = next(node for node in portfolio.descendants() if node.attrs.get("id") == "intro-title")
        assert intro.tag == "h1" and "Akshat Gupta" not in intro.text()
        bio = homepage.elements("intro-bio")[0]
        assert len([node for node in bio.descendants() if node.tag == "p"]) == 7
        bio_links = {node.attrs.get("href") for node in bio.descendants() if node.tag == "a"}
        assert {
            "https://additiv.com/", "https://www.validaitor.com/",
            "https://akshat4112.github.io/immopilot-de/", "https://github.com/Akshat4112/immopilot-de",
            "https://akshat4112.github.io/pdf-tools/", "https://github.com/Akshat4112/pdf-tools",
            "https://huggingface.co/datasets/Akshat4112/finplainlm-dpo-dataset",
            "https://github.com/Akshat4112/speaker-diffusion", f"{prefix}/posts/",
            "https://akshat4112.github.io/Glyphnet/", "https://aics.site/AICS2023/program.html",
            "https://arxiv.org/abs/2306.10392", "https://github.com/Akshat4112/Glyphnet",
            "https://huggingface.co/datasets/Akshat4112/Glyphnet",
            "https://github.com/Akshat4112/unsupervised-speaker-diarization",
            "https://github.com/Akshat4112/school-erp",
        } <= bio_links
        assert any(link and link.startswith("https://elib.uni-stuttgart.de/") for link in bio_links)
        entries = homepage.elements("work-entry")
        assert [entry.attrs.get("id") for entry in entries] == ["document-ai", "glyphnet", "pdf-tools"]
        assert [node.text() for node in portfolio.descendants()
                if node.tag == "h3"].count("GlyphNet") == 1
        links = [node.attrs.get("href") or "" for node in portfolio.descendants() if node.tag == "a"]
        assert f"{prefix}/work/" in links and f"{prefix}/about/" in links
        assert f"{prefix}/about/#{experience}" in links
        about = Document(local / "about/index.html")
        assert any(node.attrs.get("id") == experience for node in about.root.descendants())
        assert any(link.startswith("mailto:") for link in links)
        assert "https://huggingface.co/Akshat4112" in links
        assert not any("pdf-tools-home.jpg" in (node.attrs.get("src") or "")
                       for node in portfolio.descendants() if node.tag == "img")
        articles = homepage.elements("selected-writing")[0]
        article_links = [node for node in articles.descendants() if node.tag == "a"]
        assert [node.attrs.get("href") for node in article_links] == [
            "/posts/llm-agents/", "/posts/evaluating-llms/", "/posts/model-extraction-attacks/",
        ]
        if prefix:
            assert all(node.attrs.get("hreflang") == "en" and node.language == "en"
                       for node in article_links)
            assert len(homepage.elements("language-badge")) == 3
        work = Document(local / "work/index.html")
        work_links = [node.attrs.get("href") or "" for node in work.root.descendants()
                      if node.tag == "a"]
        for fragment in ("speaker-diffusion", "finplainlm-dpo-dataset", "/immopilot-de/",
                         "/nettopilot-de/", "machine-learning-case-studies", "Awesome-NLP-Resources",
                         "research-paper-presentations", "datasets/Akshat4112/Glyphnet"):
            assert any(fragment in link for link in work_links), f"{prefix}: missing {fragment}"
        assert {"/work/", "/de/work/"} <= {urlparse(link).path for link in work_links}
        assert any(node.tag == "img" and node.attrs.get("src") == "/projects/pdf-tools-home.jpg"
                   for node in work.root.descendants())
        menu = next(node for node in homepage.root.descendants() if node.attrs.get("id") == "menu")
        assert f"{prefix}/work/" in {urlparse(node.attrs.get("href") or "").path
                                     for node in menu.descendants() if node.tag == "a"}
    for path in root.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text())
        assert not any("1Qj6kaZXM40ixUgOAewhZlTj57piWERSw" in link for link in parser.links), f"Outdated CV remains linked: {path}"
    print("Bilingual homepage biography, hierarchy, Work catalogues, article links and CV verified.")


if __name__ == "__main__":
    main()
