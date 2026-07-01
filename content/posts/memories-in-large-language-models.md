---
title: "Memories in Large Language Models: How AI Models Remember and Retrieve"
date: 2025-07-10T09:00:00+01:00
draft: false
tags: ["llm", "memory", "rag", "ai", "deep-learning", "neural-networks"]
weight: 114
math: true
showtoc: true
description: "How LLMs manage memory across three time scales — context windows, KV caches, and RAG-based long-term retrieval — and what this means for building AI systems."
---

Large language models (LLMs) like GPT-4, Claude, and [Llama 3](https://github.com/meta-llama/llama) feel *almost* sentient at times. They can reference earlier parts of a conversation, recall facts from pre-training, and even “remember” user preferences across sessions. But what **is** memory in a language model?  
Is it the attention mechanism? A giant vector store? A key-value cache?  
Spoiler: it’s *all of the above*, depending on which time scale you’re talking about.

---

## Three Levels of Memory

| Time Scale | Mechanism | Typical Capacity | Example |
|------------|-----------|------------------|---------|
| **Short-Term** (ms → minutes) | **Self-attention context window** | 4K–1M tokens (GPT-4o) | Holding the current chat history |
| **Medium-Term** (minutes → hours) | **Key-Value (KV) cache, recurrent state, memory tokens** | 16K–100K tokens | ChatGPT remembering the last dozen messages in a session |
| **Long-Term** (days → years) | **External vector database, RAG, memory graphs** | Millions-billions of chunks | Notion-Q&A, enterprise knowledge bots |

---

### 1. Short-Term Memory: The Context Window

During generation, transformers perform **self-attention** over the input sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The matrices $Q$, $K$, and $V$ are computed *on the fly* for the current prompt + previous tokens (bounded by the window size). Once the window is full, older tokens are dropped or compressed (e.g., attention sink in Longformer).

> *Rule of thumb:* Context window ⟶ working memory. Nothing is stored after generation ends.

Recent research pushes this limit:

- **Anthropic Claude 3** – 200K tokens
- **GPT-4o** – 1M tokens (preview)
- **LongRoPE / RingAttention** – $O(n \log n)$ positional scaling

But linear scaling can’t keep going forever—compute and memory blow up.

### 2. Medium-Term Memory: KV Caches & Recurrent State

When serving an autoregressive model, we *cache* every $(K, V)$ pair after it is first computed. Subsequent tokens reuse these keys instead of recomputing them. This KV cache is:

- **Ephemeral** – cleared after the session ends
- **Fast** – keeps inference latency low
- **Limited** – still capped by GPU RAM (≈ 100K tokens on A100)

Some papers extend this idea:

1. **RetNet / RWKV** – blends RNN recurrence with attention, creating a compressed rolling state.
2. **Memory-Compressor** – clusters old tokens and stores centroids as *memory tokens*.
3. **[FlashAttention-2](https://arxiv.org/abs/2205.14135)** – faster block-wise attention allows longer caches.

### 3. Long-Term Memory: Retrieval-Augmented Generation (RAG)

The hot phrase of 2023-24. Instead of enlarging the transformer, **attach an external memory** ([vector database](https://en.wikipedia.org/wiki/Vector_database), [Milvus](https://milvus.io/), [Qdrant](https://qdrant.tech/), [ElasticSearch](https://www.elastic.co/)).

Pipeline:

```text
User → Embed(query) → Similarity Search → Fetch top-k docs →
      ↘︎                                ↗︎
      RAG Prompt  ←  Concatenate retrieved docs
      ↘︎
   LLM → Answer
```

Benefits:

- **Unbounded storage** – scale with cheap disk
- **Fresh knowledge** – update vectors daily without re-training
- **Auditability** – cite sources to the user

Key tricks:

1. **Hybrid search** (BM25 + dense vectors) for recall/precision balance
2. **Chunking heuristics** (overlap, window size)
3. **Self-consistency**: ask the LLM to verify retrieved facts

---

## Architectures That Combine All Three

1. **ChatGPT w/ Memory** – OpenAI stores user preferences (name, locale, facts) in a *profile store* that is re-injected at the start of new conversations.
2. **[ReAct](https://arxiv.org/abs/2210.03629) + RAG Agents** – Working memory is the agent scratch-pad; long-term memory is an external knowledge base queried on demand.
3. **Gemini + Retrieval** – Google’s long-context models combine attention over long windows with retrieval over external corpora.

The RAG flow at inference time is:

$$\text{Answer} = \text{LLM}\!\left(\,q,\; \text{Retrieve}(q, \mathcal{D}, k)\,\right)$$

where $q$ is the user query, $\mathcal{D}$ is the document store, and $k$ is the number of retrieved chunks. The retrieval step uses approximate nearest-neighbour search over dense embeddings:

$$\text{Retrieve}(q, \mathcal{D}, k) = \operatorname*{top\text{-}k}_{d \in \mathcal{D}} \cos\!\left(\phi(q),\, \phi(d)\right)$$

---

## Memory vs Privacy

Storing user interactions raises privacy flags:

- **GDPR “right to be forgotten”** – how to delete embeddings?
- **Training data leaks** – model may echo personal info seen once
- **Prompt injection** – malicious memory writes corrupt future outputs

Mitigations include *time-based TTL*, **write filters**, and **content-safety layers**.

---

## Open Challenges

1. **Catastrophic forgetting** during fine-tuning
2. **Memory editing** (e.g., remove private facts from weights) without full retraining
3. **Infinite context** research (Mixture of Sliding Windows, MEGA)
4. **Hierarchical memory** – how to decide what moves from short- to long-term

---

## The Future

In 2025-26 we’ll likely see:

- 10M-token context windows via *attention sparsity + paging*  
- *On-device* RAG for mobile agents
- **Self-organizing memory graphs** (LLM writes its own knowledge base!)
- Native support in open-source stacks like **[Mistral](https://mistral.ai/)-MoE-Memory**

The line between “model” and “database” is blurring. The most powerful systems will **amplify** LLM reasoning with structured, dynamic memory—just like the human brain combines working memory with long-term recall.

— **Akshat** 