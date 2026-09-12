# Akshat Gupta — Portfolio

Source code for [akshat4112.github.io](https://akshat4112.github.io/), the personal portfolio and technical writing site of [Akshat Gupta](https://github.com/Akshat4112), an applied AI engineer and researcher.

The site presents research, open-source projects, datasets, professional experience, and long-form writing on production AI systems.

## What the site covers

- Agentic AI and large language model systems
- Evaluation, observability, and reliability
- Retrieval-augmented generation and document intelligence
- AI security, privacy, and model extraction
- Speech processing and speaker anonymisation
- Knowledge graphs, ontologies, and vector retrieval

## Technology

- [Hugo Extended](https://gohugo.io/) 0.147.2
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) as a Git submodule
- Custom Hugo layouts and CSS
- Self-hosted [KaTeX](https://katex.org/) for mathematical notation
- GitHub Actions and GitHub Pages

## Repository structure

```text
.
├── archetypes/                 # Templates for new content
├── assets/                     # Images and Hugo-processed styles
├── config/
│   ├── _default/config.yml     # Production configuration
│   └── development/config.yml  # Local-development overrides
├── content/
│   ├── posts/                  # Technical articles
│   ├── publications/           # Research publications
│   ├── talks/                  # Talks and workshops
│   ├── events/                 # Events and community work
│   └── about.md                # Biography and experience
├── layouts/                    # Custom templates and theme overrides
├── scripts/check_content.py    # Automated article checks
├── static/                     # Files served without Hugo processing
├── themes/PaperMod/            # Theme submodule
├── PUBLISHING_CHECKLIST.md     # Article release checklist
└── WRITING_GUIDE.md            # Editorial and citation standards
```

Do not edit files inside `themes/PaperMod/` directly. Override theme behaviour with matching files under `layouts/`.

## Run locally

### Requirements

- Git
- Hugo Extended 0.147.2 or a compatible newer release
- Python 3 for content validation

### Setup

```bash
git clone --recurse-submodules https://github.com/Akshat4112/akshat4112.github.io.git
cd akshat4112.github.io
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Start the development server:

```bash
hugo server --environment development --disableFastRender
```

Open [http://localhost:1313](http://localhost:1313).

## Validate changes

Run the same content check used by the deployment workflow:

```bash
python3 scripts/check_content.py
```

Build the production site:

```bash
hugo --minify
```

For configuration or template changes, validate both environments:

```bash
hugo --minify
hugo --minify --environment development -d /tmp/akshat-portfolio-dev
```

Generated output is written to `public/` and should not be committed.

## Create content

Create a technical article:

```bash
hugo new posts/article-name.md
```

Other content types follow the same pattern:

```bash
hugo new publications/publication-name.md
hugo new talks/talk-name.md
hugo new events/event-name.md
```

Before publishing an article:

1. Follow [WRITING_GUIDE.md](WRITING_GUIDE.md).
2. Complete [PUBLISHING_CHECKLIST.md](PUBLISHING_CHECKLIST.md).
3. Run `python3 scripts/check_content.py`.
4. Run a production build and inspect the rendered page.

Article images belong under `assets/posts/`. Publication, talk, and event images should use the corresponding directory under `assets/`.

## Configuration and customisation

- Site configuration: `config/_default/config.yml`
- Development overrides: `config/development/config.yml`
- Homepage: `layouts/index.html`
- Header: `layouts/partials/header.html`
- Additional head markup: `layouts/partials/extend_head.html`
- Custom styles: `assets/css/extended/custom.css`
- Markdown image rendering: `layouts/_default/_markup/render-image.html`

The site uses British English for editorial content. Technical claims should use primary sources or official documentation, and employer work must avoid confidential project details and unsupported internal metrics.

## Deployment

A push to `main` triggers [the Hugo deployment workflow](.github/workflows/hugo.yml). The workflow:

1. checks out the repository and PaperMod submodule;
2. installs Hugo Extended 0.147.2;
3. runs the article-content validator;
4. builds the minified production site; and
5. publishes `public/` to the `gh-pages` branch.

Changes should be reviewed through pull requests before they reach `main`.

## Links

- [Live portfolio](https://akshat4112.github.io/)
- [Technical writing](https://akshat4112.github.io/posts/)
- [Research publications](https://akshat4112.github.io/publications/)
- [GitHub profile](https://github.com/Akshat4112)
- [Hugging Face](https://huggingface.co/Akshat4112)

## Licence

No repository-wide licence file is currently provided. Unless a file states otherwise, do not assume that the website source or written content is licensed for reuse.
