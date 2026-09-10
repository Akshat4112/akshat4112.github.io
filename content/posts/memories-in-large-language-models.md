---
title: "Memory in Large Language Models"
description: "A technical guide to the distinct mechanisms called LLM memory: model parameters, context, KV caches, retrieval, recurrent state, and persistent application storage."
date: 2025-07-10T09:00:00+01:00
lastmod: 2026-09-10T16:00:00+02:00
draft: false
tags: ["llm", "memory", "rag", "inference", "ai-systems"]
weight: 114
math: true
showtoc: true
---

“Memory” is an overloaded word in large language model systems. It can refer to information encoded in model parameters, tokens supplied in the current request, cached attention tensors, retrieved documents, recurrent state, or records stored by an application. These mechanisms have different lifetimes, costs, failure modes, and privacy implications.

Treating them as one feature leads to architectural mistakes. A larger context window does not create cross-session memory. A key–value (KV) cache does not make a model remember a user. Retrieval does not update model parameters. A product that recalls a preference after a week is using storage and orchestration outside the base model unless the model itself has been retrained or edited.

This article separates the mechanisms and shows how to choose between them.

## A practical taxonomy

| Mechanism | Where information lives | Typical lifetime | How it changes | Primary purpose |
|---|---|---|---|---|
| Parametric knowledge | Model weights | Across requests and deployments | Training, fine-tuning, or model editing | Generalised behaviour and knowledge |
| Input context | Tokens in the current request | One inference request | Prompt construction | Make information available to attention |
| KV cache | Per-layer key and value tensors | One generation or retained session state | Appending generated tokens | Avoid recomputing attention state |
| Recurrent state | Architecture-specific hidden or compressed state | Across segments within a run | Model-defined state transition | Carry information beyond one segment |
| Retrieval memory | Documents, records, embeddings, or a graph | Until the underlying store changes | External write and indexing pipeline | Select relevant external evidence |
| Application memory | Profiles, summaries, events, or structured facts | Across sessions | Explicit product policy | Personalisation and workflow continuity |

The first four mechanisms are part of model computation. The final two are system components. Only some of them persist after a request ends.

## 1. Parametric knowledge: information encoded in weights

Pretraining adjusts parameters $\theta$ so that the model assigns high probability to plausible continuations:

$$
\theta^* = \arg\min_\theta
\mathbb{E}_{x \sim \mathcal{D}}
\left[-\sum_t \log p_\theta(x_t \mid x_{<t})\right].
$$

The resulting weights capture statistical regularities and can reproduce some factual associations. This is often called **parametric memory**, including in the original retrieval-augmented generation formulation, which contrasts a model's parameters with an external non-parametric index ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)).

The term is useful, but weights are not a record store. There is no reliable operation such as “read row 42” or “delete this person's address.” A model may generalise, conflate facts, or fail to produce information that influenced training. Research on model editing, including ROME, studies targeted changes to factual associations, but such edits require evaluation for generalisation and unintended side effects ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)).

Parameters are suitable for stable capabilities and broad patterns. They are a poor default for frequently changing organisational facts, per-user preferences, or information that must be individually auditable and deletable.

### Memorisation is not a safe storage interface

Language models can sometimes emit sequences from their training data. Carlini et al. demonstrated extraction of verbatim examples from GPT-2, including personally identifying information present in public web data ([Carlini et al., 2020](https://arxiv.org/abs/2012.07805)). That result is a privacy risk, not a dependable memory feature. A production design should minimise sensitive training data and assess memorisation risk rather than treating weights as a user database.

## 2. Input context: information available for the current computation

The context is the token sequence supplied to a model for one inference request. It may contain system instructions, conversation history, retrieved passages, tool results, and the user's current message.

For a standard attention head,

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
$$

where $M$ contains any causal or padding mask. Tokens in the context influence the current forward pass through attention. They do not automatically persist after the serving system discards the request state.

A model's maximum context length is a capacity limit, not a guarantee that every token will be used equally well. In multi-document question answering and key–value retrieval experiments, performance was often better when relevant information appeared near the beginning or end than when it appeared in the middle ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)). Effective context therefore depends on the task, position, distractors, model, and evaluation—not only the advertised token count.

Context is appropriate when information is needed now, fits within the request budget, and should not be retained by the application.

## 3. KV caches: saved computation during autoregressive decoding

During causal generation, every new token attends to keys and values derived from preceding tokens. Recomputing those tensors for the entire prefix at each step would waste work. A KV cache retains the per-layer keys and values so the next decoding step computes them only for the new token.

For ordinary multi-head attention, a simplified cache-size estimate is

$$
\text{bytes} \approx 2 \times L \times T \times H \times b,
$$

where:

- $2$ represents keys and values;
- $L$ is the number of transformer layers;
- $T$ is the cached sequence length;
- $H$ is the hidden width represented in the cache; and
- $b$ is the number of bytes per stored element.

### Worked example

For a 32-layer model with hidden width 4,096, an 8,192-token sequence, and BF16 values ($b=2$), the simplified estimate is:

$$
2 \times 32 \times 8{,}192 \times 4{,}096 \times 2
= 4{,}294{,}967{,}296 \text{ bytes}
\approx 4 \text{ GiB}.
$$

This is for one sequence under the simplified full-width assumption. Grouped-query or multi-query attention reduces the number of cached key/value heads, while batching, parallelism, quantisation, allocator behaviour, and implementation details change the serving footprint.

| Property | Context tokens | KV cache |
|---|---|---|
| Contains user-readable text | Yes | No; it contains derived tensors |
| Changes model weights | No | No |
| Avoids repeated computation | No | Yes |
| Required after a request ends | Only if the application retains it | No, unless a serving layer deliberately reuses session state |
| Provides cross-session recall by itself | No | No |

[PagedAttention](https://arxiv.org/abs/2309.06180) addresses allocation and sharing of dynamically growing KV caches in LLM serving. [FlashAttention](https://arxiv.org/abs/2205.14135), by contrast, is an IO-aware exact attention algorithm that reduces memory traffic and avoids materialising the full attention matrix. Both improve attention efficiency; neither is a semantic or persistent memory system.

## 4. Recurrent and compressed state

Some architectures carry hidden state across segments rather than attending to every earlier token directly. Transformer-XL, for example, introduced segment-level recurrence and a relative positional scheme to extend dependency beyond a fixed segment ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)). Other designs compress, summarise, or selectively retain earlier representations.

This state should be described according to its actual contract:

- what representation is retained;
- how long it survives;
- whether gradients cross segment boundaries;
- whether the state can be inspected or edited; and
- how information loss is evaluated.

Recurrent state can extend a model's computational horizon, but it is still not equivalent to a queryable database or a product-level user profile.

## 5. Retrieval memory: external evidence selected at inference time

Retrieval-augmented generation keeps knowledge outside the model and selects relevant items for the current request. A simplified dense-retrieval pipeline is:

```text
query → encode → search external index → filter/rerank →
construct context → generate answer with evidence
```

For query encoder $\phi_q$, document encoder $\phi_d$, corpus $\mathcal{D}$, and similarity function $s$, retrieval can be written as

$$
R_k(q) = \operatorname*{top\text{-}k}_{d \in \mathcal{D}}
s\!\left(\phi_q(q), \phi_d(d)\right).
$$

The retrieved records still have to fit into the model's context or be processed through a multi-stage strategy. Retrieval therefore complements context; it does not replace it.

External retrieval is useful when information is large, changing, access-controlled, attributable, or independently deletable. Its main failure modes include poor recall, irrelevant but plausible passages, stale indexes, incorrect permissions, prompt injection in retrieved content, and answers that are not supported by the evidence.

Evaluate retrieval and generation separately:

| Layer | Example measures | Question answered |
|---|---|---|
| Retrieval | Recall@k, nDCG, filter correctness | Was the necessary evidence retrieved and ranked? |
| Context construction | Token coverage, truncation rate, duplication | Did useful evidence reach the model intact? |
| Generation | Citation correctness, faithfulness, task accuracy | Did the answer use the evidence correctly? |
| System | Latency, cost, freshness, access-control violations | Is the complete workflow acceptable in production? |

## 6. Application memory: persistent state managed by software

An assistant that recalls a preference in a later session normally relies on application-managed storage. The application decides what to write, how to represent it, when to retrieve it, and whether to place it in the next prompt. The base model does not gain a new parameter each time a user says, “I prefer concise answers.”

A minimal memory record might be:

```json
{
  "subject_id": "user-123",
  "fact": "Prefers concise technical explanations",
  "source_event": "conversation-456:message-12",
  "created_at": "2026-09-10T14:00:00Z",
  "confidence": 0.92,
  "retention_until": "2027-03-10T00:00:00Z"
}
```

The important fields are not the prose itself but its subject, provenance, time, confidence, and retention policy. A real system also needs authorisation metadata and a deletion state.

### Memory lifecycle

| Stage | Required decision | Common failure |
|---|---|---|
| Candidate extraction | Is this information useful and appropriate to retain? | Saving transient or sensitive content |
| Validation | Is it explicit, attributable, current, and permitted? | Treating an inference as a user-provided fact |
| Storage | Which subject, scope, retention period, and access policy apply? | Cross-user leakage or indefinite retention |
| Retrieval | Is it relevant to this request and allowed in this context? | Injecting stale or unrelated memories |
| Use | How should the model distinguish memory from instructions and evidence? | Prompt injection or over-trusting stored text |
| Update | Does new evidence supersede or conflict with the record? | Accumulating contradictory preferences |
| Deletion | Can the record, index entries, caches, and replicas be removed? | Deleting the source but leaving derived copies |

MemGPT illustrates one research design for moving information between a limited main context and external storage, using an operating-system-inspired hierarchy ([Packer et al., 2023](https://arxiv.org/abs/2310.08560)). The broader lesson is architectural: persistence comes from controlled data movement and storage, not from an unlimited context window.

## Privacy, retention, and deletion

Memory changes a stateless generation feature into a data-lifecycle system. Before storing conversation-derived information, define:

1. **Purpose:** which user-visible capability requires retention?
2. **Consent and expectation:** does the user know what will persist?
3. **Scope:** is the record tied to a user, organisation, agent, or workflow?
4. **Access:** which services and people can read or modify it?
5. **Provenance:** which event created the record, and was it quoted or inferred?
6. **Retention:** when does it expire or require reconfirmation?
7. **Deletion:** which primary and derived stores must be purged?
8. **Security:** can untrusted content write instructions into future prompts?

Do not promise that deleting an application record removes related information from model parameters. Conversely, retraining a model is unnecessary for deleting a preference that exists only in an application database. The mechanism determines the deletion path.

## Choosing the right mechanism

| Requirement | Start with | Reason |
|---|---|---|
| Information is needed only for the current turn | Input context | Minimal persistence and orchestration |
| Generation repeatedly extends the same prefix | KV cache | Reuses computation during decoding |
| Evidence changes frequently or needs citations | Retrieval | External content can be updated and traced |
| A preference should survive across sessions | Application memory | Explicit lifecycle and per-user control |
| Behaviour should generalise across all users | Training or fine-tuning | Changes model behaviour rather than stored records |
| Dependencies span long sequential segments | Long-context or recurrent architecture | Extends the model's computational horizon |

Most production assistants use several mechanisms together. A request may retrieve authorised records, construct a bounded context, generate with a KV cache, and then write one validated preference to persistent storage. Calling all of that “the model's memory” hides the boundaries that must be evaluated and secured.

## Practical takeaway

Ask four questions whenever a design uses the word *memory*:

1. Where exactly is the information stored?
2. What operation writes and retrieves it?
3. How long does it survive, and who can access it?
4. How is correctness, deletion, latency, and leakage tested?

If those answers are unclear, the system does not yet have a memory architecture—only a metaphor.

## References

1. Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
2. Liu, N. F. et al. (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172).
3. Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
4. Dao, T. et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
5. Dai, Z. et al. (2019). [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860).
6. Packer, C. et al. (2023). [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560).
7. Carlini, N. et al. (2020). [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805).
8. Meng, K. et al. (2022). [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262).
