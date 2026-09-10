# Technical Writing Guide

This guide defines the editorial standard for articles in Akshat Gupta's applied AI and machine learning portfolio. The goal is to show sound engineering judgement and research literacy—not to imitate a product landing page or repeat generic AI commentary.

## What every article must accomplish

A strong article should make six things clear:

1. **Problem:** What exact technical question, system behaviour, or engineering decision is being examined?
2. **Scope:** What assumptions, definitions, and boundaries apply?
3. **Approach:** How does the method or system work?
4. **Evidence:** Which equations, experiments, benchmarks, examples, or primary sources support the explanation?
5. **Limitations:** Where does the approach fail, become expensive, or depend on context?
6. **Practical takeaway:** What should an engineer, researcher, or technical decision-maker do differently after reading?

These are required functions, not mandatory section titles. Combine or rename sections when that produces a more natural narrative.

## Recommended article structure

### 1. Opening summary

Use the first two to four paragraphs to:

- state the question or claim;
- explain why it matters;
- give the reader the central conclusion;
- describe what the article will and will not cover.

Avoid scene-setting such as “AI is rapidly transforming the world,” rhetorical hype, or claims that a model is intelligent or sentient without a precise operational definition.

### 2. Problem and scope

Define the problem before presenting the solution. Identify the relevant system boundary, terminology, assumptions, and exclusions. If a term has several meanings—such as *memory*, *agent*, or *reasoning*—state which meaning is used.

### 3. Core idea and mechanism

Give the reader a compact mental model, then develop the mechanism step by step. Define every symbol before using it. Separate:

- the model or algorithm;
- the surrounding system;
- training-time behaviour;
- inference-time behaviour;
- implementation choices that are not inherent to the method.

Use diagrams only when they clarify a relationship or process. Every borrowed or adapted figure needs a source and descriptive alt text.

### 4. Evidence and implementation

Support important claims with the strongest available evidence:

- primary research papers for research claims;
- official documentation for product or API behaviour;
- reproducible experiments for empirical claims;
- clearly labelled personal observations for lessons from implementation.

Code examples should be minimal, valid, and directly connected to the argument. State benchmark conditions, datasets, baselines, and evaluation settings when reporting results.

### 5. Evaluation and trade-offs

Explain how success is measured and why those metrics are appropriate. Compare alternatives across relevant dimensions such as accuracy, latency, cost, memory, robustness, maintainability, privacy, and operational complexity.

Do not write that one method “outperforms” another unless the comparison names the task, metric, data, and conditions.

### 6. Limitations and failure modes

Include concrete limitations, not a ceremonial disclaimer. Discuss known failure modes, sensitivity to configuration, evidence gaps, deployment risks, and cases where a simpler approach is preferable.

### 7. Practical takeaway

End with a short synthesis tied to decisions or implementation. Good conclusions answer questions such as:

- When should this approach be used?
- What should be measured first?
- What is commonly misunderstood?
- What remains unresolved?

Do not use a generic “Final Thoughts” section or repeat the introduction.

### 8. References

Prefer primary sources and official documentation. Link citations where the claim appears, then provide a compact reference list when an article relies on multiple sources. For time-sensitive facts, include the applicable version or date and update the `lastmod` field.

## Three useful variations

The standard should create consistency without making every article sound identical.

- **Technical explainer:** question → intuition → formal mechanism → example → trade-offs → takeaway.
- **Applied system note:** operational problem → constraints → architecture → implementation decisions → evaluation → failure analysis → lessons.
- **Research commentary:** precise claim → prior work → evidence → competing explanations → limitations → implications.

Choose the variation that matches the article rather than forcing every heading into every post.

## Voice and style

### House voice

- Write as an applied AI engineer and researcher: direct, measured, technically curious, and specific.
- Lead with the technical substance rather than broad claims about the importance of AI.
- Prefer evidence, concrete examples, and clearly labelled personal observations over slogans.
- Use first person only for work, decisions, experiments, or observations that are genuinely yours.
- Distinguish fact, interpretation, inference, and recommendation. Signal uncertainty rather than hiding it.
- Use rhetorical questions only when one question genuinely organises the article. Avoid chains of questions as an opening device.
- Keep paragraphs focused on one idea and vary sentence length without becoming conversational or promotional.
- Replace “obviously,” “simply,” and “clearly” with the explanation the reader needs.
- Avoid “revolutionary,” “game-changing,” “cutting-edge,” and similar language. Use “state of the art” only with a dated, task-specific benchmark.

### English variant and spelling

Use British English throughout prose:

- `behaviour`, `modelling`, `optimisation`, `analyse`, `organisation`, `centre`, and `standardise`;
- `labelled` and `labelling`;
- `licence` as a noun and `license` as a verb;
- `program` for software, even in British English.

Keep the official spelling of product names, paper titles, APIs, code, commands, and direct quotations. Do not rewrite identifiers to match the house style.

Use the Oxford comma when it removes ambiguity. Use numerals for measurements, benchmark results, model sizes, token counts, dates, and versions.

### Titles and headings

- Use title case for article titles.
- Use sentence case for section and subsection headings.
- Do not use emojis in titles or headings.
- Prefer short, descriptive headings over slogans or teaser copy.
- Use a question heading only when the section directly answers it.
- Avoid terminal punctuation except for genuine questions.
- Do not use generic headings such as “Introduction”, “Deep dive”, “The future”, or “Final thoughts” when a topic-specific heading is available.
- Number headings only when the order is part of the explanation or the article is long enough to benefit from sequence markers.

### Technical terminology

Use these forms consistently:

| Preferred form | Guidance |
|---|---|
| artificial intelligence (AI) | Expand on first use when the audience may not know the abbreviation. |
| large language model (LLM) | Lower case in prose; plural is LLMs, without an apostrophe. |
| retrieval-augmented generation (RAG) | Hyphenate `retrieval-augmented`; expand on first use. |
| key-value (KV) cache | Hyphenate `key-value`; use `KV cache` afterwards. |
| fine-tune / fine-tuning | `fine-tune` is the verb; `fine-tuning` is the noun or adjective. |
| open source / open-source | Use `open source` as a noun and `open-source` before a noun. |
| machine learning | Usually unhyphenated; use `machine-learning` only when needed as a compound adjective. |
| dataset | One word. |
| pretrained | One word, following common machine-learning usage. |
| inference time / inference-time | Use the hyphenated form only before a noun, as in `inference-time cost`. |
| real time / real-time | Use `real-time` before a noun; otherwise use `in real time`. |
| ground truth / ground-truth | Use the hyphenated form only before a noun. |
| zero-shot / few-shot | Hyphenate when used as adjectives. |
| agentic AI | Lower case `agentic` unless it begins a sentence or heading. Define the concrete agent behaviour. |

Preserve the official capitalisation of model and product names. Prefer precise system terms over human analogies: write `generates`, `retrieves`, `predicts`, or `stores` instead of `thinks`, `knows`, `understands`, or `remembers` unless the article defines those terms operationally.

### Formatting mechanics

- Expand an abbreviation on its first meaningful use in each standalone article.
- Use backticks for code identifiers, commands, configuration keys, and literal values—not for emphasis.
- Italicise a new conceptual term only when introducing it; use plain text afterwards.
- Use bold sparingly for labels or distinctions, not to make whole sentences louder.
- Write link text that describes the destination; avoid “click here”.
- Introduce a list with a complete sentence and keep list items grammatically parallel.


## Citation standard

Citations should let a technically informed reader verify a claim without guessing which source supports it.

### What requires a citation

Cite the source immediately after:

- a quantitative result, benchmark, model size, cost, latency, date, or other measured value;
- a claim about what a paper introduced, demonstrated, or concluded;
- a comparison between models, methods, datasets, or systems;
- a historical attribution or priority claim;
- behaviour attributed to a product, API, library, model, or dataset;
- a time-sensitive fact, version-specific feature, or current limitation;
- a borrowed equation, diagram, table, taxonomy, code fragment, or adapted visual.

Common mathematical operations and genuinely common background knowledge do not need citations. When uncertain, cite the source.

### Source hierarchy

Use the strongest available source in this order:

1. **Primary research:** the original paper, technical report, standard, dataset paper, or benchmark specification.
2. **Official technical material:** product documentation, API references, release notes, model cards, dataset cards, or maintained repositories.
3. **Reputable secondary analysis:** surveys, textbooks, or technical articles used for context or synthesis.

Do not use Wikipedia, search-result snippets, content farms, or vendor marketing pages to support a material technical claim when a primary source exists. A vendor's official documentation is appropriate for claims about that vendor's own product, but not as neutral evidence that it is superior.

### Inline citation format

- Place the citation in the same sentence as the claim, normally at the end.
- Use descriptive linked text rather than a raw URL or an unexplained numeric marker.
- For papers, use linked author-year wording such as `[Ho et al. (2020)](https://arxiv.org/abs/2006.11239)` or integrate the linked paper title naturally into the sentence.
- For documentation, link the product or page name, for example `[PyTorch documentation](https://pytorch.org/docs/stable/)`.
- When several sources support one claim, cite only the smallest set needed and explain disagreements rather than hiding them.
- Never attach one citation to a paragraph containing several unrelated claims.

Example:

> Classifier-free guidance combines conditional and unconditional predictions without requiring a separate classifier ([Ho and Salimans, 2022](https://arxiv.org/abs/2207.12598)).

### Reference-list format

Use a `## References` section when an article relies on multiple sources. List sources in order of first appearance and use one of these formats:

- Paper: `Author(s), [Paper title](stable URL), venue or archive, year.`
- Documentation: `Organisation, [Page title](stable URL), version or date when relevant.`
- Dataset or model: `Creator or organisation, [Dataset or model name](stable URL), version or release date.`
- Software: `Maintainer or organisation, [Repository or release](stable URL), version or commit when relevant.`

Do not duplicate the same source under slightly different labels. Preserve official titles and author names.

### Stable links and versions

Prefer:

- a DOI or arXiv abstract page for a paper;
- an official documentation permalink or versioned page;
- a tagged release or commit for software behaviour tied to a version;
- an official model card, dataset card, or archival identifier;
- the original publisher or institution for standards and reports.

Remove tracking parameters. Avoid temporary share links, search-result URLs, link shorteners, and copied PDF mirrors. Add a version, release date, or access date when the source can change and the claim depends on its current state.

### Figures, tables, and code

- Use `Source: [Name](URL).` for an unchanged visual.
- Use `Adapted from [Author or source](URL).` when the presentation has been modified.
- State `Figure by the author.` when provenance would otherwise be unclear.
- Include descriptive alt text that explains the information conveyed, not merely the visual appearance.
- For reused code, link to the exact repository file, release, or commit and respect its licence.

### Verification rule

Before publishing, open every cited source and confirm that it supports the exact nearby claim. Check the relevant method, experiment, table, version, or documentation section—not only the title or abstract. If the source provides weaker or narrower evidence than the prose, narrow the claim.


## Front matter standard

Every post should include:

```yaml
---
title: "Specific, descriptive title"
description: "One sentence stating the article's scope and value."
date: 2026-01-01T09:00:00+01:00
lastmod: 2026-01-01T09:00:00+01:00
draft: true
tags: ["topic", "method"]
weight: 100
math: false
showtoc: true
cover:
    image: "/posts/example.png"
    alt: "Descriptive explanation of what the figure shows"
---
```

Remove `cover` when no meaningful visual is available. Set `math: true` only when the article contains rendered mathematics.

## Review checklist

Before setting `draft: false`, verify:

- [ ] The opening states a concrete question and central conclusion.
- [ ] The scope and important terms are explicit.
- [ ] Every material claim has a relevant citation placed in the same sentence.
- [ ] Each source directly supports the nearby claim and is primary or official where possible.
- [ ] Citation labels and reference entries follow the standard format.
- [ ] Links are stable, open correctly, and identify versions or dates when needed.
- [ ] Equations define every symbol and render correctly.
- [ ] Code is valid, focused, and necessary.
- [ ] Comparisons specify task, data, metric, and conditions.
- [ ] Limitations and failure modes are concrete.
- [ ] Figures have descriptive alt text and attribution.
- [ ] Time-sensitive facts have a date or version.
- [ ] The conclusion gives a practical takeaway without repeating the introduction.
- [ ] The title, description, tags, and `lastmod` metadata are accurate.
