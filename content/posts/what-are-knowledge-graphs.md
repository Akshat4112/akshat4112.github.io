---
title: "Knowledge Graphs: Data Models, Construction, and Graph RAG"
date: 2024-03-15T09:00:00+01:00
lastmod: 2026-09-13T00:10:00+02:00
draft: false
tags: ["knowledge-graph", "semantic-search", "ai", "data-modelling", "rag"]
weight: 107
description: "A practical guide to knowledge-graph models, identity, provenance, construction, querying, evaluation, and their role in retrieval-augmented generation."
showtoc: true
---

A knowledge graph represents entities and the relationships between them in a graph-shaped data model. Its value is not that graphs automatically contain truth or make systems explainable. The value is that identity, relationships, provenance, and domain meaning can be made explicit and queried together.

That distinction matters in applied AI. A graph built from noisy documents can preserve extraction errors. A graph query can return an irrelevant path. An ontology can support valid inference from false assertions. A language model can still misread correctly retrieved facts. Knowledge graphs create useful structure; reliability comes from the evidence, controls, and evaluation around that structure.

This article explains the major graph models, how a knowledge graph is constructed, and when graph-based retrieval is worth its additional complexity.

## What makes a graph a knowledge graph?

There is no single definition accepted across every research and engineering community. In practice, a knowledge graph usually has several of these properties:

- nodes identify entities, concepts, events, or records;
- edges represent typed relationships;
- identifiers remain stable across sources;
- a schema or ontology describes at least part of the domain;
- provenance records where claims came from;
- the graph combines information from more than one source or representation; and
- users or software can query and traverse relationships.

A formal ontology is useful but not mandatory. A small labelled-property graph with documented node and edge types can serve as a knowledge graph. An RDF graph may use RDFS or OWL to make richer semantics machine-readable. The appropriate level of formality depends on interoperability, inference, governance, and query requirements.

It is also useful to distinguish the **knowledge graph** from its storage system. Neo4j, Amazon Neptune, GraphDB, Stardog, and relational databases with graph extensions can store or query graph-shaped data. The database is infrastructure; the knowledge graph is the data, identifiers, semantics, and governance implemented on top of it.

## Two common graph data models

### RDF graphs

RDF represents information as triples:

```text
(subject, predicate, object)
```

For example, in Turtle syntax:

```turtle
:claim-1042 a :InsuranceClaim ;
    :coveredBy :policy-77 ;
    :supportedBy :document-88 .

:document-88 :containsEvidence :span-19 .
```

Subjects and predicates are identified by Internationalised Resource Identifiers (IRIs). Objects can be IRIs or literal values. RDF integrates naturally with shared vocabularies, RDFS/OWL semantics, SPARQL, and linked data.

### Labelled-property graphs

A labelled-property graph (LPG) represents nodes and relationships that can both carry labels and properties:

```cypher
(:Claim {id: "claim-1042"})
  -[:COVERED_BY]->
(:Policy {id: "policy-77"})
```

LPG systems commonly use traversal-oriented query languages such as Cypher or Gremlin. The model is convenient for application development and algorithms where properties belong directly on nodes and edges.

| Concern | RDF | Labelled-property graph |
|---|---|---|
| Basic unit | Triple | Node, relationship, and properties |
| Identity | IRIs | Database or application identifiers |
| Schema and semantics | RDFS, OWL, SHACL and vocabularies | Labels, constraints and application model |
| Common query language | SPARQL | Cypher or Gremlin |
| Interchange | Standards-oriented | Often platform-oriented, with emerging standards |
| Typical strength | Semantic integration and interoperability | Operational traversal and developer ergonomics |

This comparison is not a performance ranking. Query speed depends on the workload, indexes, engine, data shape, hardware, and query plan. Neither model is universally more flexible or more scalable.

## Identity is the foundation

The hardest part of many knowledge-graph projects is not drawing relationships. It is deciding when two records refer to the same entity.

Suppose three source systems contain:

- `ACME GmbH`, customer number `C-143`;
- `Acme Gesellschaft mit beschränkter Haftung`, registration `HRB 12345`; and
- `ACME`, named in a scanned policy document.

Merging them without evidence can connect claims or policies to the wrong organisation. Failing to merge them can fragment the entity and make queries incomplete.

Entity resolution should therefore preserve:

- source identifiers and namespaces;
- normalised attributes used for matching;
- the matching method and model version;
- a confidence or review state where appropriate;
- positive and negative evidence; and
- a way to reverse an incorrect merge.

In RDF, `owl:sameAs` asserts identity, not approximate similarity. It should not be used for a fuzzy candidate match. Weaker mapping relations or application-specific links are safer until identity is established.

Stable identifiers should be separated from human-readable labels. Names change, collide, and vary across languages. An identifier is the durable reference; labels are presentation and search metadata.

## Facts need provenance and time

A triple or edge is not self-validating. “Company A owns Company B” may be true only according to one filing, during a particular period, or under a specific interpretation.

Production graphs often need to record:

- source document or system;
- evidence location within the source;
- extraction method and model version;
- ingestion and effective timestamps;
- reviewer and review decision;
- jurisdiction or policy context; and
- confidence, where the meaning of confidence is defined.

Several representation patterns are available. RDF-star can annotate statements, named graphs can group claims by source or context, and explicit event or assertion nodes can represent provenance in either RDF or LPG systems. The best pattern is the one that supports the required queries without hiding qualification.

For an extracted claim amount, an assertion-centred model might record:

```turtle
:assertion-501 a :ExtractedAssertion ;
    :subject :claim-1042 ;
    :predicate :claimedAmount ;
    :object "42000.00"^^xsd:decimal ;
    :derivedFrom :document-88 ;
    :evidenceSpan :span-19 ;
    :producedBy :extractor-v3 ;
    :reviewStatus :Accepted .
```

This is more verbose than storing `:claim-1042 :claimedAmount 42000`, but it allows a reviewer or downstream system to inspect why the value exists.

## Schema, ontology, and validation

A graph may use several layers:

- **Vocabulary:** named classes, properties, labels, and definitions.
- **Taxonomy:** broader and narrower concept relationships.
- **Ontology:** logical semantics and relationships, often expressed with RDFS or OWL.
- **Validation shapes:** operational conformance rules, often expressed with SHACL for RDF.
- **Database constraints:** uniqueness, existence, and storage rules enforced by the graph database.

These layers are related but not interchangeable. RDFS domain and range declarations support type inference; they are not input filters. OWL commonly operates under an open-world assumption, so missing information is not automatically false. SHACL can check whether the available graph satisfies application requirements.

For a detailed explanation, see [Ontologies in Knowledge Graphs](../ontology-in-knowledge-graphs/).

## A practical construction pipeline

Imagine an insurance document-review application. It must connect policies, claims, organisations, people, documents, evidence spans, extracted assertions, and review decisions.

### 1. Define competency questions

Start with questions the graph must answer:

1. Which policy covers this claim and incident date?
2. Which document and page support the claimed amount?
3. Which extracted values were corrected by a reviewer?
4. Which organisations appear across multiple policies or claims?
5. Which required claim facts have no accepted evidence?

Competency questions bound the model and provide acceptance tests. Beginning with a generic “enterprise ontology” often produces a large graph that still cannot answer operational questions.

### 2. Inventory sources and ownership

For every source, document its owner, update cadence, identifiers, access controls, quality limitations, and deletion obligations. Decide which source is authoritative for each type of fact. Conflicting sources should be preserved and resolved explicitly rather than overwritten silently.

### 3. Design identifiers and the core model

Define namespaces and identity rules before ingestion. Keep the core vocabulary small. Model only distinctions needed for queries, validation, interoperability, or inference.

### 4. Extract candidate entities and relations

Structured sources may map directly. Documents require OCR, layout analysis, named-entity recognition, relation extraction, or language-model extraction. Preserve the original evidence and extraction metadata.

### 5. Resolve and link entities

Generate candidates using exact identifiers, blocking keys, lexical similarity, embeddings, or graph context. Evaluate candidate generation separately from the final match decision. Route ambiguous and high-impact merges to human review.

### 6. Validate and ingest

Check syntax, identifier integrity, required provenance, cardinality, datatypes, and allowed relationships. For RDF, SHACL can produce structured validation reports. Database constraints and application checks may enforce additional operational requirements.

### 7. Test queries and downstream decisions

Run competency questions against representative data, including missing, duplicated, conflicting, and time-dependent facts. Evaluate whether results support the intended decision, not merely whether the query executes.

### 8. Monitor and govern changes

Track schema versions, source changes, extraction drift, merge reversals, validation failures, query latency, and downstream incidents. A knowledge graph is a maintained product, not a one-time data migration.

## Querying the graph

In SPARQL, a query for claims and their accepted evidence might look like:

```sparql
SELECT ?claim ?amount ?document ?span
WHERE {
  ?assertion a :ExtractedAssertion ;
             :subject ?claim ;
             :predicate :claimedAmount ;
             :object ?amount ;
             :derivedFrom ?document ;
             :evidenceSpan ?span ;
             :reviewStatus :Accepted .
}
```

An equivalent LPG query depends on the chosen node-and-edge design:

```cypher
MATCH (c:Claim)<-[:SUBJECT]-(a:Assertion)-[:DERIVED_FROM]->(d:Document),
      (a)-[:SUPPORTED_BY]->(s:EvidenceSpan)
WHERE a.predicate = 'claimedAmount'
  AND a.reviewStatus = 'Accepted'
RETURN c.id, a.value, d.id, s.id
```

The graph makes the path inspectable. That is not the same as explaining why the original document contained the amount, why an extraction model selected it, or why a human decision used it. Explainability requires evidence and a claim appropriate to the mechanism being explained.

## Knowledge graphs for retrieval-augmented generation

Vector retrieval and graph retrieval solve different problems.

Vector search is effective when the query and relevant passage have similar semantic representations. Graph retrieval is useful when the answer depends on explicit identities, typed relations, constraints, or multi-hop paths. Many systems combine both.

A Graph RAG pipeline may:

1. identify entities and intent in the user query;
2. resolve those mentions to graph identifiers;
3. select relation types, filters, or a query template;
4. traverse a bounded subgraph;
5. retrieve supporting source passages for the returned assertions;
6. serialise the evidence into model context; and
7. generate an answer with citations.

For example, “Which accepted evidence supports claims filed by subsidiaries of Organisation X during the policy period?” combines identity resolution, an ownership path, a temporal constraint, review status, and evidence retrieval. A graph can express this path directly.

However, Graph RAG introduces new failure modes:

- query entities resolve to the wrong nodes;
- important entities or relations were never extracted;
- the traversal follows a valid but irrelevant path;
- stale or conflicting assertions are selected;
- graph-to-text serialisation loses qualifiers;
- the language model combines evidence incorrectly; or
- the graph answer lacks source passages that a person can verify.

Graph retrieval should therefore be compared with lexical, vector, SQL, and hybrid baselines on the actual task. It is not automatically more accurate because it is structured.

## Knowledge graphs in agent systems

An agent can use a graph as a governed tool rather than receiving an unrestricted database connection. Useful tools might expose bounded operations such as:

- resolve an entity with candidate evidence;
- retrieve assertions and provenance for one entity;
- find a typed path within a maximum depth;
- run a parameterised competency query; or
- validate a proposed assertion before write-back.

Tool schemas, access control, query limits, and audit logs matter. An agent-generated SPARQL or Cypher query can be syntactically valid while being expensive, overly broad, or semantically wrong. Read and write permissions should be separated, and high-impact graph changes should require validation or review.

## Evaluation

Evaluate the graph at component, query, and application levels.

| Layer | Example measures |
|---|---|
| Extraction | Entity and relation precision, recall, evidence-span accuracy |
| Entity resolution | Candidate recall, match precision, false-merge rate, reversal rate |
| Data quality | Validation violations, missing provenance, freshness, duplication |
| Schema and ontology | Competency-question coverage, unintended inference, maintainability |
| Query | Answer precision and recall, path relevance, temporal correctness, latency |
| Graph RAG | Retrieval recall, evidence precision, answer faithfulness, citation correctness |
| Operations | Ingestion lag, failure rate, storage growth, access-control incidents, cost |

Use a manually reviewed test set with difficult examples: aliases, near-duplicate organisations, conflicting sources, missing edges, stale facts, cyclic relationships, long paths, and temporal changes. Report performance by source and entity type instead of only one aggregate score.

For Graph RAG, separate at least four questions:

1. Was the correct entity resolved?
2. Did retrieval return the necessary assertions and source evidence?
3. Was the selected path relevant to the question?
4. Was the generated answer supported by that evidence?

A single end-to-end answer score cannot locate which layer failed.

## Common failure modes

**Graph-shaped data without identity governance.** Duplicate and wrongly merged entities corrupt every traversal built on them.

**Treating extracted statements as verified facts.** Store evidence, method, confidence, and review state. Avoid collapsing candidates and accepted assertions into the same relation.

**Forcing every source into one universal schema.** Local contexts and conflicting definitions may need mappings rather than premature unification.

**Assuming an ontology validates records.** Use validation shapes or application constraints for conformance; ontology reasoning has different semantics.

**Unbounded graph traversal.** High-degree nodes and unrestricted paths can produce irrelevant results and unpredictable cost.

**Ignoring time.** Ownership, employment, eligibility, and policy coverage change. A timeless edge can answer historical questions incorrectly.

**Claiming automatic explainability.** A path is inspectable, but its entities, relations, evidence, and inference steps must still be justified.

**No deletion and correction process.** Source deletions, privacy requests, corrected extractions, and reversed entity merges must propagate through derived data.

## When not to use a knowledge graph

A relational database may be simpler when the data is tabular, the schema is stable, transactions dominate, and queries use known joins. A vector index may be sufficient when the task is passage retrieval and explicit multi-hop relationships add little value. A document store may be preferable for independent records with flexible attributes.

Do not adopt a graph because the data contains relationships—relational systems model relationships too. A graph becomes attractive when traversal, heterogeneous integration, explicit semantics, provenance, or evolving many-to-many relationships are central enough to justify the modelling and governance cost.

## Limitations and trade-offs

Knowledge graphs require investment in identifiers, mappings, provenance, quality controls, access policies, and versioning. Entity resolution errors can have wide downstream effects. Rich graph queries can be difficult to optimise. Domain experts may disagree on definitions, and one global model may erase legitimate local differences.

Graph-based AI also inherits the limitations of its sources and extraction systems. Structure can make a false claim easier to retrieve. An inference engine derives consequences from asserted axioms; it does not independently establish truth. Human review, source verification, uncertainty handling, and application-level safeguards remain necessary.

## Practical takeaway

A useful knowledge graph is not merely a collection of nodes and edges. It is a governed representation of identity, relationships, provenance, semantics, and time that answers defined questions.

Start with competency questions and authoritative sources. Design stable identifiers. Preserve evidence and qualifications. Evaluate extraction and entity resolution before trusting traversal. Compare graph retrieval with simpler baselines. In RAG and agent systems, expose bounded, auditable graph operations and measure entity linking, path relevance, retrieval quality, and answer faithfulness separately.

The graph provides structure. Whether that structure supports reliable decisions depends on how it is built, tested, and maintained.

## References

1. Hogan, A. et al. [Knowledge Graphs](https://doi.org/10.1145/3447772). *ACM Computing Surveys*, 2021.
2. Ji, S. et al. [A Survey on Knowledge Graphs: Representation, Acquisition, and Applications](https://doi.org/10.1109/TNNLS.2021.3070843). *IEEE Transactions on Neural Networks and Learning Systems*, 2022.
3. W3C. [RDF 1.1 Concepts and Abstract Syntax](https://www.w3.org/TR/rdf11-concepts/). 2014.
4. W3C. [SPARQL 1.1 Query Language](https://www.w3.org/TR/sparql11-query/). 2013.
5. W3C. [OWL 2 Web Ontology Language: Document Overview](https://www.w3.org/TR/owl2-overview/). Second edition, 2012.
6. W3C. [Shapes Constraint Language (SHACL)](https://www.w3.org/TR/shacl/). 2017.
7. W3C. [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/). 2013.
8. Angles, R. et al. [The Property Graph Database Model](https://arxiv.org/abs/2104.13738). *AMW*, 2021.
9. Hogan, A. et al. [Knowledge Graphs and Open-World Semantics](https://kgbook.org/). 2021.
10. Edge, D. et al. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). 2024.
