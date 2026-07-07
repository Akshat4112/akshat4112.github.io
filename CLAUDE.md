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
├── config/
│   ├── _default/config.yml    # Base Hugo config (menus, params, social links)
│   └── development/config.yml # Overrides for `hugo server` (baseURL, disables GA)
├── content/                # All Markdown content (source of truth)
│   ├── posts/              # Blog posts
│   ├── publications/       # Research papers
│   ├── talks/              # Conference talks and workshops
│   ├── events/             # Events and hackathons
│   ├── about.md            # About page
│   ├── search.md           # Search page (uses PaperMod's built-in search)
│   └── archive.md          # Chronological archive page (served at /archive/)
├── layouts/                # Template overrides (take precedence over theme)
│   ├── index.html          # Custom homepage layout
│   ├── _default/_markup/
│   │   └── render-image.html  # Markdown image render hook (lazy-load + dimensions)
│   └── partials/           # Partial template overrides
│       ├── extend_head.html   # Custom <head> (author meta, KaTeX, preconnect)
│       ├── header.html        # Custom header
│       └── templates/         # OpenGraph, Twitter Cards, Schema.org overrides
├── assets/                 # Hugo Pipes-processed assets
│   ├── posts/, publications/, talks/, events/  # Cover & inline content images
│   └── akshat_gupta.jpg    # Profile photo
├── static/                 # Files served as-is (no Hugo processing)
│   ├── lib/katex/          # Self-hosted KaTeX (CSS, JS, fonts)
│   ├── favicon.ico, favicon-16x16.png, favicon-32x32.png, apple-touch-icon.png
│   └── safari-pinned-tab.svg
├── archetypes/default.md   # Template for `hugo new` content (YAML front matter)
├── themes/PaperMod/        # Theme (git submodule — do not edit directly)
└── .github/workflows/hugo.yml  # CI/CD pipeline
```

**Generated output directories** (gitignored): `public/`, `resources/_gen/`

## Content Conventions

### Front Matter (YAML)

All content uses YAML front matter (`---` delimiters). The default author is set globally in config (`params.author`), so individual posts do not need an `author` field.

```yaml
---
title: "Post Title"                  # Keep rendered <title> (title + " | Akshat Gupta") under ~65 chars
date: 2024-08-15T09:00:00+01:00      # ISO 8601 with timezone
draft: false                         # Must be false for published content
tags: ["tag1", "tag2"]               # Lowercase, hyphenated
weight: 112                          # Controls sort order (lower = higher priority)
description: "SEO description"       # Aim for 50-160 chars; used in meta tags and search
math: true                           # Enable KaTeX rendering (only set on pages using math)
showtoc: true                        # Show table of contents (lowercase) — set on all long-form posts
cover:
    image: "/section/filename.jpg"   # Path into assets/ (see Image References below)
    alt: "Descriptive alt text"
---
```

### Content Type Patterns

- **Posts** (`content/posts/`): Technical articles. Use `snake_case` or `kebab-case` filenames. Include `tags`, `description`, `weight`, `showtoc`.
- **Publications** (`content/publications/`): Academic papers. Include links to papers, author lists, abstracts.
- **Talks** (`content/talks/`): Workshop/conference talks. Include venue, date, cover images.
- **Events** (`content/events/`): Hackathons, bootcamps. Include location, date, cover images.

Each content section has an `_index.md` for the section listing page with its own SEO `description`.

### Image References

Cover images and inline content images must live in **`assets/`** (not `static/`), mirroring the content structure, e.g. `assets/posts/diagram.png`. This lets Hugo's image pipeline generate responsive `srcset` variants and explicit `width`/`height` (avoids layout shift) for cover images via the theme's `cover.html`, and for inline Markdown images via the `render-image.html` hook in this repo.

Reference them the same way regardless of physical location — Markdown images and `cover.image` both use the site-root-relative path Hugo will resolve against `assets/`:
```markdown
![Alt text](/publications/architecture.png)
```
```yaml
cover:
    image: "/events/cyber_valley.JPG"
    alt: "Descriptive alt text"
```

Only put a file in `static/` if it must be served byte-for-byte with no processing (favicons, KaTeX, files with external inbound links depending on an exact path).

## Local Development

```bash
# First-time setup: initialize theme submodule
git submodule update --init --recursive

# Start dev server (uses config/development/ overrides automatically)
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

## Key Configuration

Config lives in `config/_default/config.yml` with environment-specific overrides in `config/development/config.yml` (Hugo's directory-based environment config — a single flat `config.yml` cannot express per-environment overrides despite looking like it can with a nested `development:` key, which Hugo silently ignores).

- `baseURL`: `https://akshat4112.github.io/` in production, `http://localhost:1313/` in development
- `theme`: PaperMod
- `params.author`: `Akshat Gupta` (global default, no need to repeat in posts)
- `params.schema.sameAs`: explicit list of identity URLs for JSON-LD (LinkedIn, GitHub, Twitter, Kaggle) — do not let this fall back to `socialIcons`, which also contains the non-identity RSS link
- `profileMode`: Enabled (homepage shows profile card)
- `mainSections`: `["posts"]` — only posts appear on the homepage feed
- `outputs.home`: HTML, RSS, JSON (JSON required for search)
- Google Analytics is only active in production; the `development` config sets `googleAnalytics: ""` so local `hugo server` sessions don't pollute real analytics

## Layout Customizations

The site overrides several PaperMod partials/templates in `layouts/`:
- `layouts/index.html` — custom homepage with profile + post listing
- `layouts/partials/extend_head.html` — author meta tag, GA preconnect hint, self-hosted KaTeX loading (SRI-pinned, only on pages with `math: true`)
- `layouts/partials/header.html` — custom site header
- `layouts/partials/templates/` — custom OpenGraph, Twitter Cards, Schema.org JSON-LD
- `layouts/_default/_markup/render-image.html` — Markdown image render hook; adds `loading="lazy"` and explicit `width`/`height` (via `resources.Get`) to all inline content images

Do **not** edit files in `themes/PaperMod/` directly. Override by placing files with the same path under `layouts/`.

## Testing changes locally

Hugo is not preinstalled in every environment. Before trusting a template/config change, install Hugo (`apt-get install -y hugo` gets a close-enough extended version) and run `hugo --minify` — a clean, warning-free build with the expected page/processed-image counts is the baseline sanity check. For config changes, verify both environments explicitly, since `hugo server`'s defaults can mask a broken production build:
```bash
hugo --minify                                        # production
hugo --minify --environment development -d /tmp/dev  # development
```
