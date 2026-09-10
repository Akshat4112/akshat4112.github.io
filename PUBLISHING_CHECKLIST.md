# Publishing checklist

Use this checklist for every new or substantially revised article. The automated preflight catches structural errors; editorial and browser review still require human judgement.

## 1. Content and technical accuracy

- [ ] The opening states the question, scope, and central conclusion.
- [ ] Terminology follows `WRITING_GUIDE.md` and British English is used consistently.
- [ ] Equations define every symbol and match the surrounding explanation.
- [ ] Code examples are minimal, valid, and show their assumptions.
- [ ] Comparisons name the task, data, metric, baseline, and conditions.
- [ ] Limitations and failure cases are concrete.
- [ ] The conclusion gives a practical takeaway instead of repeating the introduction.

## 2. Evidence, claims, and confidentiality

- [ ] Every material technical or quantitative claim has a nearby primary or official source.
- [ ] Each link supports the exact nearby claim and opens without authentication.
- [ ] Figures, tables, equations, and reused code have appropriate attribution.
- [ ] Illustrative values are labelled; measured results state their source and evaluation context.
- [ ] Personal work is supported by a public artefact or confirmed by the author.
- [ ] Employer and client material contains no private names, data, prompts, metrics, architecture, or incidents.

## 3. Metadata and chronology

- [ ] `title`, `description`, `date`, `draft`, `tags`, `weight`, and `showtoc` are present.
- [ ] `description` is a single useful sentence and does not repeat the title.
- [ ] `lastmod` is present for a substantial revision and is not earlier than `date`.
- [ ] `math: true` is present when display or inline mathematics is used.
- [ ] The publication and modification dates reflect the real history of the article.
- [ ] Tags use the existing lowercase, hyphenated vocabulary.

## 4. Links, media, and rendering

- [ ] Internal links resolve and external links use stable canonical URLs.
- [ ] Local images exist, load, and are not blurred or stretched.
- [ ] Every image has descriptive alt text and a source when it is not original.
- [ ] Tables remain understandable on a narrow screen.
- [ ] Code blocks identify the language and do not overflow unreadably.
- [ ] Mathematics renders without raw delimiters or clipped equations.
- [ ] Headings produce a useful table of contents with no skipped levels.

## 5. Final browser review

Check the built article at approximately 390 px and 1440 px widths.

- [ ] Title, date, reading metadata, cover, and table of contents appear correctly.
- [ ] There is no horizontal page overflow at mobile width.
- [ ] Paragraphs, lists, tables, code, images, and equations have readable spacing.
- [ ] Light and dark themes preserve contrast and syntax highlighting.
- [ ] Keyboard focus is visible on links and controls.
- [ ] The previous/next navigation and all article-specific links work.

## 6. Release commands

Run these checks from the repository root:

```bash
python3 scripts/check_content.py
hugo --minify
```

Publish only when the automated preflight and Hugo build pass and the browser review has no unresolved blocking issue.
