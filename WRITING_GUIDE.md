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

## Voice and wording

- Write in clear, direct English with an engineering and research perspective.
- Prefer specific nouns and verbs over promotional adjectives.
- Use first person only for work, decisions, or observations that are genuinely yours.
- Distinguish fact, interpretation, and recommendation.
- Expand acronyms on first use.
- Keep paragraphs focused on one idea.
- Replace “obviously,” “simply,” and “clearly” with the missing explanation.
- Avoid “revolutionary,” “game-changing,” “state of the art,” and similar wording unless rigorously supported.

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
- [ ] Technical claims are correct and supported by primary sources where possible.
- [ ] Equations define every symbol and render correctly.
- [ ] Code is valid, focused, and necessary.
- [ ] Comparisons specify task, data, metric, and conditions.
- [ ] Limitations and failure modes are concrete.
- [ ] Figures have descriptive alt text and attribution.
- [ ] Time-sensitive facts have a date or version.
- [ ] The conclusion gives a practical takeaway without repeating the introduction.
- [ ] The title, description, tags, and `lastmod` metadata are accurate.
