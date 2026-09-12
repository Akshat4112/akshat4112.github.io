---
title: "Vector Databases: Indexes, Retrieval, and Production Trade-offs"
date: 2023-12-15T09:00:00+01:00
lastmod: 2026-09-13T01:10:00+02:00
draft: false
tags: ["vector-databases", "semantic-search", "embeddings", "rag", "ai-infrastructure"]
weight: 103
description: "A guide to vector search, ANN indexes, filtering, hybrid retrieval, production operations, and deciding whether a dedicated vector database is necessary."
showtoc: true
math: true
---

A vector database stores vectors alongside identifiers and metadata, indexes them for similarity search, and provides database capabilities such as persistence, filtering, updates, access patterns, and operational management. Its purpose is broader than running a nearest-neighbour algorithm.

That distinction separates a database from a library such as FAISS. FAISS provides efficient vector indexing and search primitives. It does not, by itself, provide the complete persistence, multi-tenancy, metadata, authorisation, backup, replication, and service-management layer expected from a production database.

A dedicated vector database is also not always necessary. PostgreSQL with `pgvector`, Elasticsearch or OpenSearch, a multimodal database, or an in-process index can be the better choice when it reduces operational complexity while meeting retrieval requirements.

The engineering question is not “Which vector database is best?” It is “Which retrieval architecture meets the required relevance, filtering correctness, freshness, latency, isolation, deletion, availability, and cost on representative data?”

## From content to vectors

An embedding model maps an input $x$—such as a passage, image, product, or audio segment—to a vector:

\[
f(x)=\mathbf{v}\in\mathbb{R}^{d}.
\]

The geometry is learned from the model's training objective. Similarity is meaningful only relative to that model, its input preparation, and the task. A 1,536-dimensional vector is not inherently more semantic than a 384-dimensional vector, and two embedding models cannot normally share one index unless their vector spaces are explicitly compatible.

For text retrieval, the indexing pipeline typically:

1. parses and cleans documents;
2. divides them into retrievable units;
3. preserves document, section, page, and access-control metadata;
4. creates an embedding for each unit;
5. stores the vector and source reference; and
6. builds or updates a search index.

The query passes through the same compatible embedding model. Search returns nearby vectors, but the application should resolve them back to the authoritative source content. Treat the index as a retrieval structure, not the system of record.

## Distance and similarity measures

Common measures include cosine similarity, inner product, and Euclidean distance.

Cosine similarity compares direction:

\[
\operatorname{cos}(\mathbf{x},\mathbf{y})=
\frac{\mathbf{x}^{\top}\mathbf{y}}
{\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2}.
\]

Squared Euclidean distance compares position:

\[
d^2(\mathbf{x},\mathbf{y})=
\lVert\mathbf{x}-\mathbf{y}\rVert_2^2.
\]

Inner product is:

\[
s(\mathbf{x},\mathbf{y})=\mathbf{x}^{\top}\mathbf{y}.
\]

If all vectors are L2-normalised, ranking by cosine similarity, inner product, and squared Euclidean distance is equivalent. Without normalisation, magnitude affects inner product and Euclidean distance differently.

Use the metric for which the embedding model was trained or documented. Benchmarking multiple index metrics cannot repair an embedding model that does not represent the task-relevant notion of similarity.

## Exact and approximate nearest-neighbour search

Given query vector $\mathbf{q}$ and $N$ stored vectors, exact search computes the distance to every candidate and returns the closest $k$. A flat index provides the ground truth for evaluating approximate search and can be entirely practical for small collections or hardware-accelerated workloads.

Approximate nearest-neighbour (ANN) search reduces work by accepting that it may miss some exact neighbours. The quality of an ANN index is commonly measured using recall against exact search:

\[
\operatorname{Recall@}k=
\frac{|A_k(\mathbf{q})\cap E_k(\mathbf{q})|}{k},
\]

where $A_k$ is the approximate top-$k$ set and $E_k$ is the exact top-$k$ set.

ANN recall is not retrieval relevance. An index can reproduce the embedding model's exact neighbours perfectly while those neighbours are irrelevant to users. Measure both:

- **index recall:** agreement with exact vector search; and
- **task relevance:** agreement with human-labelled relevant results or downstream outcomes.

## HNSW

Hierarchical Navigable Small World (HNSW) builds a layered proximity graph. Search begins in sparse upper layers, moves greedily towards the query, and refines candidates in denser lower layers.

Important controls include:

- `M`: approximate number of graph connections per element;
- `efConstruction`: candidate-list size during index construction; and
- `efSearch`: candidate-list size during querying.

Increasing these parameters usually improves index recall at the cost of memory, build time, or query latency. HNSW is attractive for high recall and interactive search, but its graph consumes substantial memory. Inserts are supported by many implementations; deletions, compaction, and filtered search behaviour depend on the database.

## IVF and product quantisation

An inverted-file (IVF) index partitions the vector space into coarse clusters. At query time, it searches the nearest `nprobe` clusters instead of the entire collection.

Increasing `nprobe` generally improves recall and increases latency. The index must be trained on representative vectors; a poor or stale partition can reduce search quality when the distribution changes.

Product quantisation (PQ) compresses vectors by splitting them into subvectors and replacing each subvector with a codebook entry. Distances can be approximated using compact codes rather than full-precision vectors.

IVF and PQ are often combined:

- IVF reduces the number of candidates searched;
- PQ reduces memory and distance-computation cost; and
- optional reranking uses stored full-precision vectors for the best candidates.

Compression can materially improve capacity and throughput, but may reduce index recall. Measure the effect by query type, not only on an average benchmark.

| Index | Main idea | Typical strength | Main trade-off |
|---|---|---|---|
| Flat | Compare every vector | Exact reference and simple updates | Cost grows with collection size |
| HNSW | Traverse a proximity graph | High-recall interactive search | Memory and construction cost |
| IVF | Search selected coarse clusters | Controllable speed–recall balance | Requires representative training and probing |
| PQ | Search compressed codes | Lower memory footprint | Approximation error |
| IVF-PQ | Partition and compress | Large-scale compact search | More parameters and quality tuning |

Index names are not performance guarantees. Implementation, hardware, concurrency, filtering, and data distribution can dominate algorithm-level expectations.

## Filtering changes the retrieval problem

Production searches rarely ask only for the nearest vectors. They also require constraints:

- tenant or user access;
- document type and language;
- jurisdiction or business unit;
- validity period;
- ingestion or review status; and
- source or confidentiality classification.

Filtering can occur before, during, or after ANN search.

**Pre-filtering** restricts candidates before vector search. It preserves filter correctness but may leave too few candidates or require specialised filtered indexes.

**Integrated filtering** applies constraints while traversing the index. Its performance and recall depend on the engine and filter selectivity.

**Post-filtering** retrieves vector candidates first and removes disallowed records afterwards. It can return fewer than $k$ results and must never be used as the sole security boundary if unauthorised content can enter application memory or logs.

Access control should be enforced in the retrieval service and underlying data system, not expressed only in a language-model prompt. Test adversarial queries, low-selectivity filters, high-selectivity filters, and empty-result behaviour.

## Hybrid retrieval

Dense embeddings are good at semantic similarity but can miss exact identifiers, rare names, codes, dates, and literal phrases. Lexical methods such as BM25 are strong on token evidence but may miss paraphrases.

Hybrid retrieval combines candidate lists from dense and lexical search. Common strategies include:

- weighted score fusion after normalising incomparable score ranges;
- Reciprocal Rank Fusion (RRF), which combines ranks rather than raw scores; and
- unioning candidates before a learned reranker.

RRF assigns a document score such as:

\[
\operatorname{RRF}(d)=
\sum_{r\in R}\frac{1}{c+\operatorname{rank}_{r}(d)},
\]

where $R$ contains the retrievers and $c$ limits the influence of very high ranks.

Hybrid retrieval is not automatically better. It adds index, tuning, and failure-analysis complexity. Evaluate dense, lexical, and hybrid systems on slices such as semantic questions, exact identifiers, multilingual queries, and out-of-domain language.

## A worked RAG retrieval example

Consider an insurance document assistant that answers questions about policy coverage. The source corpus contains policy wording, schedules, endorsements, and claim documents from multiple tenants.

### Ingestion

Each retrievable passage stores:

```json
{
  "chunk_id": "policy-77:end-4:p2:c3",
  "document_id": "endorsement-4",
  "tenant_id": "tenant-18",
  "policy_id": "policy-77",
  "document_type": "endorsement",
  "valid_from": "2025-05-01",
  "valid_to": null,
  "page": 2,
  "embedding_model": "embedding-model-v3",
  "content_hash": "..."
}
```

The authoritative text remains in document storage. The vector index contains a source reference and enough metadata for retrieval and policy enforcement.

### Query

For “Was flood damage covered on 20 June 2025?”, the retriever might:

1. authenticate the user and derive the permitted tenant and policies;
2. parse the relevant date and policy identifier;
3. pre-filter by tenant, policy, and validity interval;
4. retrieve dense and lexical candidates;
5. fuse or rerank the candidates;
6. diversify overlapping chunks;
7. return passages with document and page provenance; and
8. require the generator to cite only returned evidence.

The date filter and endorsement relationship may matter more than vector similarity. A semantically similar exclusion from the wrong policy or time period is a dangerous retrieval error.

### Evaluation

For each labelled question, record the passages necessary to answer it. Then measure:

- Recall@k: whether required evidence appears in the candidate set;
- Precision@k: how much retrieved context is relevant;
- MRR or nDCG: whether relevant evidence is ranked early;
- metadata-filter correctness and unauthorised-result count;
- answer faithfulness and citation correctness;
- no-answer behaviour when evidence is absent;
- p50, p95, and p99 latency; and
- embedding, storage, reranking, and generation cost.

Tune retrieval before judging generation. If required evidence is absent, the language model cannot reliably recover it.

## Database, library, and integrated vector support

### Similarity-search libraries

FAISS and similar libraries provide algorithms and data structures for vector search. They are useful for research, offline processing, embedded services, and custom systems. The application owns persistence, metadata, distributed serving, replication, access control, and lifecycle management.

### Dedicated vector databases

Dedicated systems focus on vector ingestion, ANN indexes, metadata filtering, distributed search, and vector-oriented APIs. Managed offerings can reduce infrastructure work. They also introduce another data system, consistency model, pricing model, security surface, and operational dependency.

### Relational databases with vector extensions

PostgreSQL with `pgvector` can keep vectors, relational metadata, transactions, and access rules together. This can be compelling when the corpus already lives in PostgreSQL and its scale and latency are sufficient. Query planning, index settings, connection management, and vacuum or update behaviour still require measurement.

### Search engines with vector support

Elasticsearch and OpenSearch combine lexical retrieval, vector search, structured filters, aggregation, and mature search operations. They are natural candidates for hybrid retrieval when the organisation already operates them. They may be heavier than necessary for small applications.

| Situation | Reasonable first candidate |
|---|---|
| Small prototype or offline experiment | Exact NumPy/PyTorch search or FAISS |
| Existing PostgreSQL application and moderate corpus | `pgvector` |
| Existing search platform and hybrid text requirements | Elasticsearch or OpenSearch |
| Vector-first workload with large scale or managed-service needs | Dedicated vector database |
| Tight control over algorithms and hardware | Custom service using a search library |

This is a starting point, not a vendor selection rule. Benchmark with the expected corpus, filters, update pattern, and concurrency.

## Updates, versioning, and deletion

Vectors become stale when source content, chunking, or the embedding model changes. Store the source version, content hash, chunker version, and embedding-model version for each record.

Prefer versioned re-indexing over mixing incompatible embeddings. A safe migration can:

1. build a new index namespace;
2. backfill from authoritative content;
3. compare retrieval on a fixed evaluation set;
4. shadow production queries;
5. switch reads when acceptance criteria pass; and
6. retain a rollback window before deleting the old index.

Deletion must propagate from the source of truth to chunks, vectors, caches, replicas, backups, and derived indexes according to policy. Tombstones alone may not meet a requirement for physical deletion. Test deletion by searching for the removed content and inspecting asynchronous queues or failed jobs.

Frequent updates can degrade some ANN structures or create fragmentation. Measure insert latency, time until searchable, delete visibility, compaction behaviour, and recall after sustained churn.

## Multi-tenancy and security

Multi-tenant retrieval must prevent cross-tenant results at every layer. Common architectures use separate indexes, namespaces, partitions, or mandatory metadata filters. The correct choice depends on tenant count, size distribution, isolation requirements, and operating cost.

Security controls should include:

- authentication and authorisation before query execution;
- server-controlled tenant filters;
- encryption and secret management;
- network isolation where required;
- audit logs for reads, writes, and administration;
- least-privilege service identities;
- protection against sensitive content in logs and traces; and
- tested backup, restore, retention, and deletion procedures.

Embeddings are not anonymisation. They can preserve sensitive information and may be vulnerable to inference or reconstruction attacks. Apply the same data classification and access governance used for the source content.

## Benchmarking retrieval and operations

A production benchmark needs more than queries per second.

| Dimension | Measures |
|---|---|
| Relevance | Recall@k, Precision@k, MRR, nDCG, reranker gain |
| ANN fidelity | Recall against exact search by query slice |
| Filtering | Constraint correctness, result count, recall under filter selectivity |
| Latency | p50/p95/p99 for embedding, search, reranking, and total retrieval |
| Throughput | Queries and updates per second at realistic concurrency |
| Freshness | Ingestion-to-searchable delay and update visibility |
| Reliability | Error rate, timeout rate, failover and restore time |
| Isolation | Cross-tenant leakage tests and authorisation failures |
| Lifecycle | Re-index duration, delete propagation, compaction impact |
| Cost | Embedding, storage, memory, compute, network, licence, and operations |

Build the relevance set from real information needs. Include exact identifiers, paraphrases, ambiguous queries, unsupported questions, multilingual content, access-restricted documents, recent updates, and hard negatives that share vocabulary but answer a different question.

Run load tests with realistic filters and top-$k$ values. A benchmark without metadata filtering can misrepresent both latency and recall. Measure cold and warm behaviour, batch ingestion, background compaction, and failure recovery.

## Common failure modes

**Choosing the database before building an evaluation set.** Vendor benchmarks cannot identify which passages are relevant to your users.

**Treating FAISS as a complete database.** A library can power the search layer, but the application must supply the surrounding database and operational guarantees.

**Assuming every search is approximate.** Exact search is a useful baseline and can be sufficient at modest scale.

**Using vector search for exact identifiers.** Lexical search, structured lookup, or filters are usually more reliable for invoice numbers, policy IDs, dates, and codes.

**Post-filtering security-sensitive results.** Filtering after retrieval can underfill results and expose data to intermediate components.

**Mixing embedding versions.** Distances across incompatible spaces are meaningless.

**Evaluating only answer quality.** An LLM can occasionally answer correctly despite poor retrieval. Measure retrieval and generation separately.

**Deleting only the database row.** Derived vectors, caches, replicas, and backups need an explicit lifecycle.

**Ignoring zero-result behaviour.** The system should distinguish “no authorised evidence found” from infrastructure failure and avoid filling the gap with unsupported generation.

## When a vector database is unnecessary

Do not add a vector database if simpler retrieval meets the requirement. Examples include:

- the corpus is small enough for exact in-memory search;
- queries are primarily identifiers, filters, or structured joins;
- PostgreSQL or an existing search engine meets relevance and latency targets;
- the content changes so rapidly that indexing cost exceeds its value;
- a curated set of rules or navigation paths answers the task reliably; or
- there is no labelled retrieval evaluation showing that embeddings add value.

A vector database cannot repair poor chunking, missing source data, unsuitable embeddings, weak access control, or an undefined notion of relevance.

## Practical takeaway

Start with an exact-search and lexical baseline. Build a labelled retrieval set before selecting infrastructure. Choose an embedding model and distance measure together. Add ANN only when corpus size or latency requires it, and tune HNSW, IVF, or PQ against both exact-search recall and task relevance.

Treat filters as part of retrieval and authorisation. Compare dense, lexical, and hybrid approaches. Version embeddings and indexes, make deletion observable, test tenant isolation, and measure tail latency and full operating cost.

The best vector-search architecture is the simplest one that satisfies the complete production contract—not the one with the most specialised label.

## References

1. Johnson, J., Douze, M. and Jégou, H. [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734). *IEEE Transactions on Big Data*, 2019.
2. Malkov, Y. A. and Yashunin, D. A. [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320). *IEEE TPAMI*, 2020.
3. Jégou, H., Douze, M. and Schmid, C. [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57). *IEEE TPAMI*, 2011.
4. Karpukhin, V. et al. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906). *EMNLP*, 2020.
5. Robertson, S. and Zaragoza, H. [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). *Foundations and Trends in Information Retrieval*, 2009.
6. Cormack, G. V., Clarke, C. L. A. and Buettcher, S. [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114). *SIGIR*, 2009.
7. Meta. [FAISS documentation](https://faiss.ai/).
8. pgvector. [Open-source vector similarity search for PostgreSQL](https://github.com/pgvector/pgvector).
9. Elasticsearch. [Dense vector field type](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
10. OpenSearch. [k-NN vector search documentation](https://docs.opensearch.org/latest/search-plugins/knn/).
