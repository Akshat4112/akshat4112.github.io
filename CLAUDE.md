# CLAUDE.md

## Project Overview

Personal portfolio website for Akshat Gupta, hosted at [akshat4112.github.io](https://akshat4112.github.io/). Built with **Hugo** (static site generator) using the **PaperMod** theme, deployed to **GitHub Pages** via GitHub Actions.

## Tech Stack

- **Hugo Extended** v0.147.2+ — static site generator
- **PaperMod theme** — included as a git submodule at `themes/PaperMod`
- **KaTeX** v0.16.8 — math rendering, self-hosted in `static/lib/katex/`
- **GitHub Actions** — CI/CD via `.github/workflows/hugo.yml`, deploys to `gh-pages` branch on push to `main`

## Repository Structure

```
├── config.yml              # Hugo site configuration (menus, params, social links)
├── content/                # All Markdown content (source of truth)
│   ├── posts/              # Blog posts (14 articles on ML/AI topics)
│   ├── publications/       # Research papers
│   ├── talks/              # Conference talks and workshops
│   ├── events/             # Events and hackathons
│   ├── about.md            # About page
│   ├── search.md           # Search page (uses PaperMod's built-in search)
│   └── archive.md          # Archive page
├── layouts/                # Template overrides (take precedence over theme)
│   ├── index.html          # Custom homepage layout
│   └── partials/           # Partial template overrides
│       ├── extend_head.html   # Custom <head> (author meta, KaTeX)
│       ├── header.html        # Custom header
│       └── templates/         # OpenGraph, Twitter Cards, Schema.org overrides
├── static/                 # Static assets
│   ├── posts/              # Post images
│   ├── publications/       # Publication images
│   ├── talks/              # Talk images
│   ├── events/             # Event images
│   ├── lib/katex/          # Self-hosted KaTeX (CSS, JS, fonts)
│   ├── favicon.ico         # Favicons (generated from profile image)
│   ├── favicon-16x16.png
│   ├── favicon-32x32.png
│   └── apple-touch-icon.png
├── assets/                 # Hugo-processed assets (profile photo)
├── archetypes/default.md   # Template for `hugo new` content (YAML front matter)
├── themes/PaperMod/        # Theme (git submodule — do not edit directly)
└── .github/workflows/hugo.yml  # CI/CD pipeline
```

**Generated output directories** (gitignored): `public/`, `resources/_gen/`

## Content Conventions

### Front Matter (YAML)

All content uses YAML front matter (`---` delimiters). The default author is set globally in `config.yml` (`params.author`), so individual posts do not need an `author` field.

```yaml
---
title: "Post Title"
date: 2024-08-15T09:00:00+01:00    # ISO 8601 with timezone
draft: false                         # Must be false for published content
tags: ["tag1", "tag2"]               # Lowercase, hyphenated
weight: 112                          # Controls sort order (lower = higher priority)
description: "SEO description"       # Used in meta tags and search
math: true                           # Enable KaTeX rendering
showtoc: true                        # Show table of contents (lowercase)
cover:
    image: "/section/filename.jpg"   # Cover image path (relative to static/)
---
```

### Content Type Patterns

- **Posts** (`content/posts/`): Technical articles. Use `snake_case` or `kebab-case` filenames. Include `tags`, `description`, `weight`.
- **Publications** (`content/publications/`): Academic papers. Include links to papers, author lists, abstracts.
- **Talks** (`content/talks/`): Workshop/conference talks. Include venue, date, cover images.
- **Events** (`content/events/`): Hackathons, bootcamps. Include location, date, cover images.

Each content section has an `_index.md` for the section listing page.

### Image References

Images are stored in `static/` mirroring the content structure. Reference them in Markdown with absolute paths from site root:
```markdown
![Alt text](/publications/architecture.png)
```

Cover images in front matter also use this convention:
```yaml
cover:
    image: "/events/cyber_valley.JPG"
```

## Local Development

```bash
# First-time setup: initialize theme submodule
git submodule update --init --recursive

# Start dev server
hugo server --bind 0.0.0.0 --baseURL http://localhost:1313 --disableFastRender

# Create new content
hugo new posts/my-new-post.md
hugo new publications/my-publication.md
hugo new talks/my-talk.md
hugo new events/my-event.md

# Build for production
hugo --minify
```

## Deployment

Push to `main` triggers the GitHub Actions workflow (`.github/workflows/hugo.yml`):
1. Checks out repo with submodules
2. Builds with `hugo --minify` (Hugo Extended v0.147.2)
3. Deploys `public/` to `gh-pages` branch via `peaceiris/actions-gh-pages@v4`

## Key Configuration (`config.yml`)

- `baseURL`: `https://akshat4112.github.io/`
- `theme`: PaperMod
- `params.author`: `Akshat Gupta` (global default, no need to repeat in posts)
- `profileMode`: Enabled (homepage shows profile card)
- `mainSections`: `["posts"]` — only posts appear on the homepage feed
- `outputs.home`: HTML, RSS, JSON (JSON required for search)
- Math rendering enabled globally (`params.math: true`)
- Google Analytics: `G-8YC2E5MW2M` (disabled in dev via `development` block)

## Layout Customizations

The site overrides several PaperMod partials in `layouts/`:
- `layouts/index.html` — custom homepage with profile + post listing
- `layouts/partials/extend_head.html` — author meta tag + self-hosted KaTeX loading
- `layouts/partials/header.html` — custom site header
- `layouts/partials/templates/` — custom OpenGraph, Twitter Cards, Schema.org JSON-LD

Do **not** edit files in `themes/PaperMod/` directly. Override by placing files with the same path under `layouts/`.
