---
title: "Ontologies in Knowledge Graphs: Semantics, Reasoning, and Validation"
date: 2024-01-15T09:00:00+01:00
lastmod: 2026-09-12T23:30:00+02:00
draft: false
tags: ["ontology", "knowledge-graph", "semantic-web", "ai", "data-modelling"]
weight: 104
description: "A practical guide to ontology design in knowledge graphs, including RDF, RDFS, OWL, open-world reasoning, SHACL validation, and production trade-offs."
showtoc: true
math: true
---

An ontology gives a knowledge graph an explicit, machine-readable account of a domain: the concepts that exist, the relationships between them, and the statements that follow from those definitions. It can align data from different systems, support inference, and make the intended meaning of a graph inspectable.

That does not make an ontology a database schema, a data-quality checker, or a guarantee that every statement in the graph is true. Those responsibilities require different mechanisms. In particular, OWL reasoning and SHACL validation answer different questions:

- **OWL asks what can be inferred** from the asserted facts and axioms.
- **SHACL asks whether a data graph conforms** to specified constraints.
- **SPARQL asks which graph patterns match** the stored or inferred data.

Keeping these roles separate is essential when knowledge graphs are used in search, document intelligence, retrieval-augmented generation (RAG), or regulated decision-support systems.

## Scope and terminology

The word *ontology* is used at several levels of formality. A lightweight ontology might contain only a controlled vocabulary, a class hierarchy, and named relationships. A more expressive ontology can define logical restrictions, equivalence, disjointness, and property characteristics that a reasoner can use to derive new statements or detect logical inconsistency.

Related artefacts serve different purposes:

| Artefact | Primary purpose | Typical examples |
|---|---|---|
| Taxonomy | Arrange concepts in a hierarchy | `Invoice` is a subclass of `FinancialDocument` |
| Thesaurus | Relate preferred terms, synonyms, and broader or narrower concepts | SKOS concept schemes |
| Ontology | Define domain semantics and logical relationships | OWL classes, properties, and restrictions |
| Validation schema | Test data against operational constraints | SHACL shapes |
| Database schema | Define stored structure and database constraints | SQL tables, keys, and `NOT NULL` constraints |

A knowledge graph can exist without a formal ontology. Conversely, an ontology can be developed before any instance data is loaded. The value of an ontology depends on the application: a simple graph may need only stable identifiers and a small vocabulary, while cross-system integration or automated reasoning may justify a richer model.

## The RDF data model

RDF represents information as triples:

\[
(\text{subject},\ \text{predicate},\ \text{object})
\]

For example:

```turtle
:claim-1042 a :InsuranceClaim ;
    :reportedBy :broker-17 ;
    :supportedBy :document-88 .
```

Each resource and property should have a stable identifier, normally an Internationalised Resource Identifier (IRI). The triple structure is deliberately simple. Most domain meaning comes from the vocabulary layered on top of it.

RDFS can define basic class and property semantics:

```turtle
:InsuranceClaim a rdfs:Class .
:Broker a rdfs:Class .

:reportedBy a rdf:Property ;
    rdfs:domain :InsuranceClaim ;
    rdfs:range :Broker .
```

The `rdfs:domain` declaration does not mean “reject any subject that is not already labelled as an insurance claim”. It means that a subject using `:reportedBy` can be inferred to be an `:InsuranceClaim`. Similarly, the range statement permits inference that the object is a `:Broker`.

This inference-oriented interpretation is one reason RDF vocabularies must not be treated as if they were relational constraints.

## What OWL adds

OWL provides more expressive constructors for describing a domain. Common examples include:

- class equivalence and disjointness;
- intersections, unions, and complements;
- object and datatype property restrictions;
- inverse, transitive, symmetric, and functional properties;
- minimum, maximum, and exact cardinalities;
- identity statements such as `owl:sameAs`.

Suppose a document-review application defines a reviewed claim as a claim with at least one review decision:

```turtle
:ReviewedClaim a owl:Class ;
    owl:equivalentClass [
        a owl:Class ;
        owl:intersectionOf (
            :InsuranceClaim
            [
                a owl:Restriction ;
                owl:onProperty :hasReviewDecision ;
                owl:minCardinality "1"^^xsd:nonNegativeInteger
            ]
        )
    ] .
```

Given an `:InsuranceClaim` with a review decision, an OWL reasoner can classify it as a `:ReviewedClaim`. This is classification, not procedural code: the ontology states a logical condition, and the reasoner derives what follows from it.

Expressivity has a cost. Rich axioms can be harder for teams to understand, more expensive to reason over, and easier to misuse. OWL therefore defines profiles such as OWL 2 EL, QL, and RL for common computational settings. Choosing the most expressive language available is rarely a good default; choose the least expressive model that supports the required inferences.

## Open-world reasoning changes the meaning of missing data

OWL generally uses the **open-world assumption**: if a fact is absent, it is not automatically false. The graph may simply be incomplete.

Consider this statement:

```turtle
:claim-1042 a :InsuranceClaim .
```

If no policyholder is recorded, an OWL reasoner does not conclude that the claim has no policyholder. It concludes only that the graph does not currently state one. This differs from the closed-world behaviour expected in many databases and application validations.

OWL also does not use the **unique-name assumption** by default. Two different IRIs are not necessarily two different entities unless their difference is asserted or follows from other axioms. That matters when cardinality restrictions and entity resolution interact.

These semantics are useful for integrating incomplete information, but surprising when developers expect missing fields to trigger errors. Operational requirements such as “every submitted claim must have exactly one claim number” belong in a validation layer.

## SHACL is for graph validation

SHACL defines shapes against which an RDF data graph can be validated. The following shape requires each insurance claim to have exactly one string-valued claim number and at least one supporting document:

```turtle
:InsuranceClaimShape a sh:NodeShape ;
    sh:targetClass :InsuranceClaim ;
    sh:property [
        sh:path :claimNumber ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype xsd:string
    ] ;
    sh:property [
        sh:path :supportedBy ;
        sh:minCount 1 ;
        sh:class :Document
    ] .
```

A SHACL processor evaluates target nodes and produces a validation report. Depending on the application, a violation can block ingestion, create a review task, or be recorded as a warning.

| Requirement | Appropriate mechanism |
|---|---|
| Infer a class from an existing relationship | RDFS or OWL reasoning |
| Detect an impossible combination of logical axioms | OWL consistency checking |
| Require a property before accepting a record | SHACL validation |
| Restrict a field to an operational pattern or datatype | SHACL validation |
| Retrieve claims connected to a document | SPARQL query |
| Enforce transaction and storage constraints | Database or application layer |

OWL restrictions can resemble validation constraints syntactically, but their semantics differ. Under an open-world interpretation, an existential restriction can imply that some value exists without requiring that the value be explicitly present in the current data graph. SHACL evaluates the available graph against a conformance rule.

## A worked modelling example

Imagine a document-intelligence system that extracts claim facts from PDFs. The system must connect a claim, policy, claimant, broker, supporting documents, extracted evidence, and review decisions.

A useful modelling process begins with **competency questions**—questions the graph must answer:

1. Which source documents support a claim amount?
2. Which extracted values were accepted, corrected, or rejected by a reviewer?
3. Which policy was active on the incident date?
4. Which model and prompt version produced an extraction?
5. Which claims lack sufficient supporting evidence?

These questions identify the required concepts and relations more reliably than beginning with a large class diagram. A small vocabulary might include:

```turtle
:Extraction a owl:Class .
:EvidenceSpan a owl:Class .
:ReviewDecision a owl:Class .

:derivedFrom a owl:ObjectProperty ;
    rdfs:domain :Extraction ;
    rdfs:range :Document .

:supportedBySpan a owl:ObjectProperty ;
    rdfs:domain :Extraction ;
    rdfs:range :EvidenceSpan .

:reviewedAs a owl:ObjectProperty ;
    rdfs:domain :Extraction ;
    rdfs:range :ReviewDecision .
```

The ontology provides shared semantics. SHACL shapes can then require provenance fields for every production extraction. SPARQL queries can retrieve the evidence behind a decision. Application code still controls authorisation, transactions, user workflow, and model execution.

This separation produces a more auditable system than storing only the final extracted value. It does not make the result correct by itself: the source document may be wrong, the extraction may be wrong, or two identifiers may have been incorrectly reconciled.

## Designing an ontology in practice

### 1. Start with decisions and queries

Document the users, decisions, and competency questions first. If no application needs a distinction, adding it may create maintenance cost without practical value.

### 2. Reuse established vocabularies carefully

Reuse can improve interoperability and reduce duplicated modelling. Examples include RDF, RDFS, OWL, SKOS, PROV-O for provenance, and domain vocabularies. Reuse only terms whose published semantics match the intended meaning; identical labels do not imply identical concepts.

### 3. Separate identifiers from labels

Labels change across languages and organisations. Stable IRIs should identify concepts and entities, while human-readable labels are annotations. Avoid using a display name as the identity key.

### 4. Model provenance explicitly

For AI-assisted systems, record where a claim came from, when it was generated, which system produced it, and whether a human reviewed it. PROV-O or a smaller application-specific provenance model can support this without asserting that every generated statement is trustworthy.

### 5. Keep inference intentional

For every axiom, write down the inference it is meant to enable. Domain and range declarations, transitive properties, inverse properties, and identity links can produce consequences across the graph. Test them on representative and adversarial examples.

### 6. Define validation separately

Translate ingestion and application contracts into SHACL shapes. Decide which violations are errors, warnings, or informational findings, and establish whether validation occurs before or after entailment.

### 7. Version and test the model

Ontology changes can alter query results and inferred types even when instance data is unchanged. Version IRIs and release notes where appropriate, keep competency-question tests, and run regression checks for reasoning, SHACL reports, and important SPARQL queries.

## Ontologies in RAG and agent systems

An ontology can help an AI application by providing controlled entity types, relation names, query constraints, and links between synonymous concepts. In graph-based RAG, it may improve retrieval by making multi-hop relationships explicit. In tool-using agents, it can describe the domain objects that tools accept and return.

The ontology does not guarantee grounded generation. A system can retrieve the wrong subgraph, misidentify an entity, traverse an irrelevant relation, or generate a statement unsupported by the retrieved evidence. Evaluation should therefore test the entire pipeline:

- entity linking and disambiguation;
- graph retrieval recall and precision;
- path relevance;
- answer faithfulness to retrieved evidence;
- provenance coverage;
- latency and reasoning cost;
- behaviour when the graph is incomplete or contradictory.

For many applications, a small ontology plus strict validation and good provenance is more useful than a highly expressive ontology that few contributors can maintain.

## Common failure modes

**Treating OWL as a form validator.** Missing properties may not be logical errors under open-world semantics. Use SHACL for conformance checks.

**Using `owl:sameAs` for approximate matches.** `owl:sameAs` states identity, so every property of one resource can apply to the other. Use a weaker mapping relation when two records or concepts are merely similar.

**Overloading domain and range.** These declarations infer types; they are not input filters. Unexpected data can therefore lead to unexpected classifications.

**Building the model without competency questions.** A large ontology can look comprehensive while failing to answer the queries the application actually needs.

**Ignoring provenance and time.** Facts may depend on a source, jurisdiction, model version, or effective date. Flattening these qualifications into timeless triples can make the graph misleading.

**Assuming consistency means truth.** A logically consistent graph can still contain false assertions. Reasoning preserves the consequences of the model and its data; it does not independently verify reality.

## Evaluation and operational checks

Ontology quality cannot be reduced to one score. A production review should combine logical, task, data, and operational evidence.

| Area | Questions to test |
|---|---|
| Logical consistency | Do representative datasets produce unintended contradictions? |
| Competency questions | Can the required questions be answered with stable SPARQL queries? |
| Inference precision | Are inferred classes and relations expected and useful? |
| Validation | Do SHACL shapes catch known malformed and incomplete records? |
| Coverage | Which required domain concepts and source fields remain unmapped? |
| Maintainability | Can domain experts understand definitions and review changes? |
| Performance | Are reasoning, validation, and query latency acceptable at realistic scale? |
| Governance | Are ownership, versioning, provenance, and deprecation rules explicit? |

Evaluate with examples from the real data distribution, including missing fields, duplicate entities, conflicting sources, unexpected types, and version changes. Synthetic happy-path triples alone will not reveal the most expensive failures.

## Limitations and trade-offs

Ontologies introduce governance and maintenance work. Teams must agree on identifiers and definitions, map source systems, manage versions, and investigate unintended inferences. Reasoning performance varies with ontology expressivity, data size, engine, and workload. Distributed organisations may not agree on one universal conceptual model.

Not every graph needs OWL. A labelled-property graph with application-level validation may be sufficient for a bounded use case. RDF and OWL become more compelling when shared semantics, interoperability, inference, or standards-based tooling provide concrete value. The architecture should follow the decision being supported, not the prestige of the modelling technology.

## Practical takeaway

An ontology is a semantic contract: it makes the intended concepts and logical relationships of a domain explicit. In a robust knowledge-graph application, use RDFS or OWL for meaning and inference, SHACL for data conformance, SPARQL for retrieval, and database or application controls for operational enforcement.

Start small, design from competency questions, preserve provenance, and test the consequences of every important axiom. The goal is not the richest possible ontology. It is a model that helps people and systems interpret, validate, retrieve, and govern knowledge reliably.

## References

1. W3C. [RDF 1.1 Concepts and Abstract Syntax](https://www.w3.org/TR/rdf11-concepts/). 2014.
2. W3C. [RDF Schema 1.1](https://www.w3.org/TR/rdf-schema/). 2014.
3. W3C. [OWL 2 Web Ontology Language: Structural Specification and Functional-Style Syntax](https://www.w3.org/TR/owl2-syntax/). Second edition, 2012.
4. W3C. [OWL 2 Web Ontology Language: Profiles](https://www.w3.org/TR/owl2-profiles/). Second edition, 2012.
5. W3C. [OWL 2 Web Ontology Language: Direct Semantics](https://www.w3.org/TR/owl2-direct-semantics/). Second edition, 2012.
6. W3C. [Shapes Constraint Language (SHACL)](https://www.w3.org/TR/shacl/). 2017.
7. W3C. [SPARQL 1.1 Query Language](https://www.w3.org/TR/sparql11-query/). 2013.
8. W3C. [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/). 2013.
9. Noy, N. F. and McGuinness, D. L. [Ontology Development 101: A Guide to Creating Your First Ontology](https://protege.stanford.edu/publications/ontology_development/ontology101.pdf). Stanford Knowledge Systems Laboratory, 2001.
