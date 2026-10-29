# Portfolio Refresh Implementation Plan

> **For agentic workers:** Use Superpowers execution and review guidance; keep content, templates, and verification scoped to this refresh.

**Goal:** Show current public work and correct bilingual profile details under the website update approved in chat.

**Architecture:** Retain Hugo and the pinned PaperMod theme. Update English and German content, add a public PDF Tools engineering case study, and override the writing section's ordering without changing theme files.

**Tech stack:** Existing Hugo Extended, Go templates, CSS, and dependency-free Python checks.

**Spec:** The five approved updates in chat: project highlights, Now, About, public engineering evidence, and writing chronology.

## Constraints

- Use public repository evidence for project claims; label ImmoPilot unreleased and NettoPilot in development.
- Retain official employment title and dates; use the owner's stated Stuttgart work location and current stack.
- Keep English and German discovery paths consistent; clearly label the English case study from German pages.
- Publish no employer implementation details or invented outcome metrics.
- Add no runtime dependencies; preserve existing layout and article pagination.

## Tasks

- [x] Update `content/about*.md` and `content/now*.md` in a delegated content task; preserve original publication dates and update modification dates.
- [x] Add an actual product screenshot, bilingual homepage project highlights, and links to development projects in `layouts/index.html`.
- [x] Add `content/posts/pdf-tools-output-validation.md` using dated, immutable public evidence and explicit limitations.
- [x] Override `layouts/posts/list.en.html` to paginate by descending date; retain German chronology and curated homepage writing.
- [x] Run content validation, production/development builds, all rendered checks, and mobile/desktop browser review.
- [x] Obtain independent review and resolve findings; include validation evidence in the pull request.

Independent review found only missing About modification dates; both were corrected.
Local production and development builds used the CI-pinned Hugo Extended 0.147.2.
Responsive review covered 1440 px desktop and 390 px mobile, with no page overflow or missing images.

## Review focus

- Development projects must not appear as released applications.
- English pagination and the German archive must expose the same complete date order.
- Screenshot, case-study links, language routes, and dates must resolve correctly.
- Mobile screenshots and tables must not cause horizontal page overflow.
- Article claims must remain within the public audit's documented scope.
