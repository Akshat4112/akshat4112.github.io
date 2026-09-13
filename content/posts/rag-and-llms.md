---
title: "Building Reliable RAG Systems"
date: 2024-07-15T09:00:00+01:00
lastmod: 2026-09-13T01:10:00+02:00
draft: false
tags: ["rag", "llm", "retrieval", "document-intelligence", "production-ai"]
weight: 111
math: true
showtoc: true
description: "An end-to-end guide to building and evaluating production RAG systems, from document ingestion and hybrid retrieval to grounded generation and observability."
---

Retrieval-augmented generation (RAG) gives a language model access to external evidence at inference time. It is useful when answers depend on private, changing, or domain-specific information that should not be encoded only in model weights.

RAG does not make an answer correct by construction. A system can retrieve the wrong document, omit a decisive table row, use an obsolete version, misunderstand accurate evidence, or attach a citation that does not support its claim. Retrieval changes the failure surface; it does not remove it.

The original RAG work combined a sequence-to-sequence model's parametric memory with dense, non-parametric retrieval ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)). In production applications, the idea becomes a larger information system:

\[
\text{source} \rightarrow \text{parse} \rightarrow \text{index} \rightarrow
\text{retrieve} \rightarrow \text{rerank} \rightarrow \text{generate} \rightarrow
\text{verify}.
\]

This article follows that complete path for document-heavy applications.

## Define the answer contract first

Before choosing an embedding model or vector store, define what a valid answer must do.

For a policy assistant, the contract might require the system to:

- use only documents the requester is authorised to access;
- prefer the currently effective policy version;
- distinguish quoted evidence from interpretation;
- cite the exact page or section behind each material claim;
- abstain when the available evidence is missing or contradictory; and
- meet a stated latency and cost budget.

This contract determines ingestion metadata, retrieval filters, prompt design, evaluation labels, and monitoring. Without it, teams often optimise retrieval relevance while leaving versioning, access control, or citation correctness unspecified.

## The offline pipeline: turn documents into evidence

Most RAG failures begin before the user's query arrives. The offline pipeline decides what information can later be found and cited.

### 1. Acquire and version sources

Record the source URI, document identifier, version, effective date, ingestion time, owner, access policy, and content checksum. Treat updates as new versions rather than silently overwriting indexed content.

The source of truth should remain recoverable. An embedding is not an audit record, and extracted text may lose layout, tables, footnotes, or reading order.

### 2. Parse structure, not only text

For digital documents, preserve headings, lists, tables, page numbers, captions, and links. For scans, retain OCR confidence and coordinates alongside the recognised text. These fields support filtering, citation display, and failure analysis.

A flat text dump is especially risky for financial statements, forms, and policies. A table cell may be meaningless without its row label, column label, units, and reporting period. Chunking should therefore follow semantic and layout boundaries where possible.

### 3. Create retrievable units

There is no universally correct chunk size. Small chunks can retrieve a precise sentence but omit the surrounding qualification. Large chunks preserve context but reduce retrieval precision and consume more tokens.

A practical unit can contain:

- the section heading path;
- the local paragraph or table;
- a small amount of neighbouring context;
- source and version metadata; and
- a stable citation anchor.

Evaluate chunking with real questions. Character counts such as “500 words with overlap” are starting parameters, not design conclusions.

### 4. Build more than one retrieval representation

Dense embeddings help match semantically similar language. Sparse lexical retrieval remains useful for exact product names, policy codes, identifiers, dates, and uncommon terms. A production index often stores both, together with metadata for permissions, dates, document types, and versions.

FAISS is a library for efficient vector similarity search, not a complete database. It does not by itself provide the document lifecycle, access control, backups, filtering semantics, replication, or operational guarantees expected from a production data system.

## The online pipeline: retrieve usable context

At query time, the goal is not to find text that looks similar. It is to assemble the smallest authorised evidence set that can answer the request.

### 1. Interpret and constrain the request

Normalise obvious spelling or formatting differences, but keep the original query for traceability. Derive filters only from trusted application state—for example tenant, permissions, locale, effective date, and product—not from instructions embedded in retrieved documents.

Some questions need decomposition. “Which exclusions changed between the 2024 and 2025 policies?” requires retrieving two versions and aligning comparable sections. A single embedding search may not represent that task well.

### 2. Retrieve candidates with complementary methods

Dense retrieval maps a query \(q\) and document chunk \(d\) into vectors and scores their similarity, often with a dot product:

\[
s_{\text{dense}}(q,d) = E_q(q)^\top E_d(d).
\]

Dense Passage Retrieval showed that learned dense retrieval can outperform a strong sparse baseline on several open-domain question-answering datasets ([Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)). That does not imply dense retrieval always wins on private documents. Exact identifiers, rare terms, and newly introduced names often favour lexical matching.

A hybrid system runs dense and sparse retrieval, then fuses their ranked lists. Reciprocal rank fusion is a simple option:

\[
\operatorname{RRF}(d)=\sum_{r \in R}\frac{1}{k+\operatorname{rank}_r(d)},
\]

where \(R\) contains the retrieval methods and \(k\) limits the influence of very high ranks. Tune fusion and candidate counts against retrieval labels rather than assuming equal weights.

### 3. Rerank for the actual question

First-stage retrieval should favour recall: collect a manageable candidate pool that is unlikely to omit the evidence. A reranker then spends more computation estimating query–passage relevance and reduces the pool passed to the generator.

Cross-encoders jointly process the query and candidate, while late-interaction approaches such as ColBERT retain token-level interactions with more reusable document representations ([Santhanam et al., 2021](https://arxiv.org/abs/2112.01488)). The appropriate choice depends on corpus size, latency, hardware, and the value of improved ranking.

Reranking cannot recover evidence that the first stage never retrieved. Measure both candidate recall and post-reranking quality.

### 4. Assemble context deliberately

Remove exact duplicates, keep source diversity where the task requires it, and avoid filling the context window merely because space is available. Relevant evidence can be displaced or diluted by redundant passages.

Ordering matters as well. Long-context models can use evidence inconsistently depending on where it appears; “Lost in the Middle” found lower performance when relevant information was placed in the middle of long inputs ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)). Test context length and ordering with the target model rather than treating a larger window as a substitute for retrieval.

## Generate answers that expose their evidence

The generation prompt should state the evidence policy, output contract, and abstention behaviour. For example:

~~~text
Answer only from the supplied passages.
For each material claim, cite the passage identifier.
If the passages do not support an answer, say what is missing.
If passages conflict, describe the conflict instead of choosing silently.
Return the required schema without additional fields.
~~~

The application should pass passage identifiers separately from document text and map them to trusted citation metadata after generation. Do not allow retrieved text to invent a source identifier or destination URL.

Structured output helps verify that each claim has citations, but schema validity does not prove that the citations support the claim. Citation correctness needs an entailment check or human review on representative samples.

### Abstention is a product behaviour

“I do not know” is not automatically correct. The system may abstain despite adequate evidence or answer confidently when evidence is absent. Evaluate both:

- **answerable recall**: how often answerable questions receive a useful answer; and
- **unanswerable precision**: how often abstentions occur only when the evidence is insufficient.

For high-risk tasks, route uncertain or conflicting cases to a reviewer rather than asking the model to resolve uncertainty through confident prose.

## A worked document example

Suppose a user asks:

> Does the current commercial policy cover water damage caused by gradual leakage?

The corpus contains:

- a current policy wording;
- a superseded wording with a different exclusion;
- a broker summary;
- an endorsement that overrides one section; and
- scanned claim guidance with a low-confidence OCR passage.

A reliable pipeline should:

1. apply tenant, product, and effective-date filters;
2. retrieve “gradual leakage”, related water-damage language, and the relevant exclusion;
3. retrieve endorsements linked to the current policy;
4. rerank policy wording and endorsements above the broker summary;
5. pass the decisive sections with version and page metadata;
6. require the answer to distinguish base wording from the endorsement; and
7. cite both sources or abstain if their relationship cannot be established.

A semantically plausible answer based on the superseded wording is still a failure. This is why document version, relationship, and authority belong in the retrieval contract.

## Evaluate components and the complete system

An end-to-end answer score is necessary, but it cannot identify where a regression began. Evaluate each boundary.

### Retrieval evaluation

Create relevance labels at the passage or evidence-set level. Measure:

- Recall@\(k\): whether the required evidence appears in the candidate set;
- Precision@\(k\): how much of the retrieved set is relevant;
- mean reciprocal rank for tasks with a decisive passage;
- nDCG when relevance has graded levels;
- version and permission-filter accuracy; and
- evidence coverage for questions requiring multiple passages.

Slice results by document type, OCR quality, query type, language, table presence, and corpus freshness.

### Generation evaluation

Given a fixed evidence set, measure:

- claim-level faithfulness to the supplied passages;
- answer correctness and completeness;
- citation correctness and coverage;
- handling of conflicts;
- calibrated abstention; and
- output-schema validity.

RAGAS proposed automated metrics that separate retrieval relevance, faithfulness, and answer quality ([Es et al., 2023](https://arxiv.org/abs/2309.15217)). Model-based metrics can accelerate iteration, but calibrate them against human judgements in the application domain before using them as release gates.

### End-to-end evaluation

Run the complete pipeline on typical, boundary, critical, and known-regression cases. Track task success alongside safety, latency, token consumption, and cost per successful answer.

Use controlled ablations to locate value:

| Comparison | Question answered |
|---|---|
| Dense vs sparse vs hybrid | Which candidate strategy improves evidence recall? |
| With vs without reranking | Does ranking quality justify its latency? |
| Current chunking vs alternative | Are boundaries losing necessary context? |
| Retrieved context vs oracle evidence | Is the main limit retrieval or generation? |
| Candidate vs production baseline | Does the change improve the whole system? |

The oracle-evidence test is particularly useful. If generation fails even when supplied with the correct passages, changing the retriever is unlikely to fix the problem.

## Observe production RAG without storing everything blindly

Every request should be reproducible enough to investigate while respecting privacy and retention rules. Useful trace fields include:

- query and access-policy version;
- parser, chunker, embedding, and index versions;
- retrieval method, filters, candidate scores, and ranks;
- reranker version and selected passages;
- prompt and model configuration identifiers;
- response, citations, abstention decision, and evaluator results;
- stage-level latency, token use, and cost; and
- user feedback linked to a trace identifier.

Do not log restricted document content by default. Store identifiers, hashes, or redacted excerpts when full text is unnecessary, and apply the same authorisation and retention controls to observability data as to source documents.

Monitor distributions, not only averages. A stable mean Recall@10 can hide a collapse for scanned tables or a new policy version. Alerts should connect to actionable slices and recent pipeline changes.

## Failure modes and responses

| Failure mode | Diagnostic signal | Typical response |
|---|---|---|
| Wrong document version | High semantic relevance, invalid effective date | Enforce metadata filters and version relationships |
| Exact identifier missed | Dense miss, sparse hit | Add lexical retrieval or query normalisation |
| Relevant passage omitted | Low candidate Recall@\(k\) | Revisit parsing, chunking, embeddings, or candidate count |
| Relevant passage ranked too low | Good candidate recall, weak nDCG/MRR | Tune fusion or add a reranker |
| Correct evidence, unsupported claim | Retrieval passes, faithfulness fails | Tighten answer contract and claim verification |
| Excessive context harms answer | Quality falls as context grows | Deduplicate, rerank, and reduce context |
| Permission leak | Retrieved source outside access set | Apply authorisation before retrieval and test invariants |
| Stale index | Source version newer than indexed version | Add freshness checks and ingestion reconciliation |

RAG should reduce the opportunity for unsupported generation when relevant evidence is available and used correctly. It cannot prevent hallucination, guarantee freshness, or enforce permissions without explicit mechanisms and tests.

## Where Self-RAG fits

Self-RAG is not simply query rewriting. It trains a language model to retrieve passages adaptively and to generate special reflection tokens that assess retrieval need, passage relevance, response support, and response utility ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)).

That is a specific learned architecture, not a label for every pipeline with iterative search or self-critique. In many production systems, simpler application-level routing, retrieval thresholds, verification, and abstention may be easier to inspect and govern. Use Self-RAG when its learned adaptive behaviour matches the task and can be evaluated against a simpler baseline.

## RAG, fine-tuning, and tools solve different problems

Use RAG when the answer depends on inspectable external knowledge that changes independently of the model. Use fine-tuning when the goal is to change behaviour, format, terminology, or task performance through training examples. Use deterministic tools or APIs for authoritative calculations, transactions, and structured records.

These approaches can be combined, but none substitutes for a clear system boundary:

- a fine-tuned model can still use stale facts;
- RAG can retrieve authoritative facts yet misinterpret them; and
- a tool can return the right data while an agent calls it with the wrong arguments.

Choose the mechanism based on the source of truth and the failure you need to control.

## Practical takeaway

A production RAG system is an evidence pipeline, not a vector-search demo. Reliability depends on versioned ingestion, structure-aware parsing, complementary retrieval, measured reranking, deliberate context assembly, grounded generation, calibrated evaluation, and traceable operations.

The central question is not “Did the model receive some relevant text?” It is:

> Did the system retrieve the authorised and decisive evidence, use it faithfully, expose its sources, and behave safely when the evidence was insufficient?

## References

1. Lewis, P. et al. (2020). [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401). arXiv:2005.11401.
2. Karpukhin, V. et al. (2020). [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906). arXiv:2004.04906.
3. Santhanam, K. et al. (2021). [*ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*](https://arxiv.org/abs/2112.01488). arXiv:2112.01488.
4. Liu, N. F. et al. (2023). [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172). arXiv:2307.03172.
5. Es, S. et al. (2023). [*RAGAS: Automated Evaluation of Retrieval Augmented Generation*](https://arxiv.org/abs/2309.15217). arXiv:2309.15217.
6. Asai, A. et al. (2023). [*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*](https://arxiv.org/abs/2310.11511). arXiv:2310.11511.
