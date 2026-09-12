---
title: "Understanding Attention in Transformers"
date: 2024-08-15T09:00:00+01:00
draft: false
tags: ["transformers", "attention", "deep-learning", "nlp", "self-attention", "neural-networks", "ai"]
weight: 112
math: true
showtoc: true
description: "A concise technical guide to transformer attention: masks, position, efficient kernels, KV caches, and interpretability limits."
---

Attention lets a model construct a new representation of each token by mixing information from other tokens. The mixing weights depend on the current input, so the same word can use different context in different sentences.

That description is useful but incomplete. Attention does not inherently know token order. A causal mask is not a positional encoding. A key-value (KV) cache is not long-term memory. Retrieval-augmented generation (RAG) and mixture-of-experts (MoE) are not attention variants. Attention maps can also be informative without being faithful explanations of a prediction.

This article separates those concepts and follows one attention operation from its tensor shapes to production inference.

## Scaled dot-product attention

For one attention head, let

$$
Q \in \mathbb{R}^{L_q \times d_h}, \qquad
K \in \mathbb{R}^{L_k \times d_h}, \qquad
V \in \mathbb{R}^{L_k \times d_v}.
$$

The operation introduced in the original transformer is

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V,
$$

where $M$ is an optional mask. The matrix $QK^\top$ contains one score for every allowed query-key pair. Division by $\sqrt{d_h}$ controls the score scale as the head dimension grows. Softmax normalises each query's scores across keys, and the result forms a weighted sum of the value vectors.

Common implementation shapes are:

| Tensor | Shape | Meaning |
|---|---|---|
| $Q$ | $[B,H,L_q,d_h]$ | Queries for each batch item and head |
| $K$ | $[B,H_{kv},L_k,d_h]$ | Keys, possibly shared across query heads |
| $V$ | $[B,H_{kv},L_k,d_v]$ | Values, possibly shared across query heads |
| Scores | $[B,H,L_q,L_k]$ | Pairwise compatibility before softmax |
| Output | $[B,H,L_q,d_v]$ | Contextualised representation per head |

For standard multi-head attention, $H_{kv}=H$. Multi-query and grouped-query attention use fewer KV heads.

### A small numerical example

Suppose one query produces scaled scores $[1.2, 0.3, -0.4]$. Softmax gives approximately $[0.62, 0.25, 0.13]$, so the output is

$$
0.62V_1 + 0.25V_2 + 0.13V_3.
$$

This is a soft mixture, not a hard lookup. A large weight says that a value contributes strongly at this operation; it does not prove that its source token caused the final output.

## Multi-head attention

A model normally projects the hidden states into several heads:

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),
$$

$$
\operatorname{MHA}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O.
$$

Heads provide separate learned projection spaces. Some heads correlate with syntactic or positional patterns, but it is unsafe to claim that every head learns one clean, human-readable relation. Behaviour is distributed across heads, layers, residual streams and feed-forward blocks.

## Attention types are defined by inputs and masks

Terms often grouped as “types of attention” describe different design axes.

### Self-attention

Queries, keys and values are derived from the same sequence. Encoder-style models usually allow each token to attend bidirectionally to all non-padding tokens. Decoder-only language models use self-attention with a causal mask.

### Cross-attention

Queries come from one sequence while keys and values come from another. In an encoder-decoder model, decoder states query the encoder output. The same pattern can connect text to image or audio features.

### Causal attention

Causal attention prevents position $t$ from reading positions later than $t$:

$$
M_{t,j}=
\begin{cases}
0 & j \le t,\\
-\infty & j > t.
\end{cases}
$$

After softmax, disallowed positions receive zero probability. Causality is a visibility constraint; it does not encode the distance or order between visible tokens.

### Local and sparse attention

Dense attention represents all $L_qL_k$ query-key interactions. Local attention restricts each query to a window, reducing the interaction count to roughly $O(Lw)$ for window size $w$. Sparse schemes combine patterns such as local windows, global tokens, blocks or random connections. Longformer and BigBird are examples.

The practical benefit depends on kernels that exploit the pattern. Applying a sparse mask to a dense kernel may not produce the expected speed or memory reduction.

### Approximate and efficient exact attention

Performer approximates softmax attention so computation can be reassociated without constructing the full score matrix. FlashAttention does something different: it computes exact softmax attention, up to numerical precision, while reducing transfers between high-bandwidth and on-chip memory through tiled computation.

| Design | Changes allowed pairs? | Dense result | Main objective |
|---|---:|---:|---|
| Causal mask | Yes | Exact for the mask | Prevent future-token access |
| Local or sparse attention | Yes | Different | Reduce pairwise interactions |
| Kernel approximation | Usually no explicit sparsity | Approximate | Avoid the full score matrix |
| FlashAttention | No | Exact | Improve memory traffic and kernel efficiency |

## Position is supplied separately

Without positional information, unmasked self-attention is permutation equivariant: reordering the input reorders the outputs in the same way. Token content alone does not indicate which token came first or how far apart two tokens are.

Architectures therefore add position through a separate mechanism, including:

- learned or sinusoidal vectors added to token embeddings;
- relative position biases added to attention scores;
- rotary position embeddings (RoPE), which rotate query and key components; or
- linear attention biases such as ALiBi.

These choices affect length generalisation and distance representation. RoPE and ALiBi are positional mechanisms, not attention types.

## Attention inside a transformer block

A transformer is more than attention. A modern block combines attention with residual connections, normalisation and a feed-forward network. The order varies: the original transformer used post-normalisation, while many large language models use pre-normalisation.

A simplified pre-normalised decoder block is:

```text
x = x + causal_self_attention(norm(x))
x = x + feed_forward(norm(x))
```

Encoder-decoder models may insert cross-attention between self-attention and the feed-forward network. Architectures vary widely in depth, width, head count, normalisation, activation and positional design, so a fixed layer-count range is not generally meaningful.

RAG and MoE belong elsewhere:

- **RAG** retrieves external evidence and supplies it to a generator. The model may process that evidence with ordinary self- or cross-attention.
- **MoE** routes tokens to expert networks, commonly in the feed-forward sublayer. Routing is not an attention variant.

## KV caching during autoregressive inference

During training, causal attention for many positions can be evaluated in parallel. During generation, the model produces one token at a time. Recomputing keys and values for the full prefix at every step would repeat substantial work.

A KV cache stores keys and values already computed at each layer. The next query attends to the cached prefix plus the new token. This reduces repeated computation, but cache memory grows with context length.

Ignoring implementation overhead, storage is approximately

$$
2 \times N_{layers} \times B \times L \times H_{kv} \times d_h \times s,
$$

where two represents keys and values and $s$ is bytes per element. The cache is inference state for one sequence—not learned memory, a retrieval store or a way to remember previous conversations after the cache is discarded.

### MHA, MQA and GQA

- **Multi-head attention (MHA):** every query head has its own KV head.
- **Multi-query attention (MQA):** all query heads share one KV head.
- **Grouped-query attention (GQA):** groups of query heads share KV heads.

MQA and GQA reduce cache size and memory bandwidth. The choice depends on model quality, serving hardware, batch size and latency targets.

## An implementation sketch

Production code should use an optimised framework primitive, but this sketch exposes the main operations:

```python
import math
import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    # q: [batch, heads, query_length, head_dim]
    # k, v: [batch, heads, key_length, head_dim]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    if mask is not None:
        # mask is True where attention is allowed
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return weights @ v
```

Real implementations must handle padding, causal masking, grouped KV heads, numerical stability, mixed precision, dropout and fused kernels. PyTorch's `scaled_dot_product_attention` can select an optimised backend and is preferable to a handwritten production kernel.

## Attention weights are not explanations by default

Attention maps can reveal structure, catch masking errors and support exploratory analysis. They are not automatically faithful explanations.

Jain and Wallace showed that different attention distributions can sometimes produce similar predictions and that weights may correlate poorly with gradient-based importance. Wiegreffe and Pinter argued that the conclusion depends on the explanation claim and evaluation protocol. The practical lesson is that faithfulness must be tested rather than assumed.

For a stronger interpretability study:

1. state what the visualisation is claimed to explain;
2. compare it with perturbation, counterfactual or causal interventions;
3. test whether changing high-weight tokens changes the output as predicted;
4. report stability across examples, heads, layers and seeds; and
5. avoid presenting one heatmap as a general explanation.

## Practical evaluation

Evaluate an attention design as part of the whole model and serving system.

| Concern | What to measure |
|---|---|
| Task quality | Target metrics and difficult sequence-length slices |
| Training | Tokens per second, peak memory and convergence |
| Prefill | Latency and memory across prompt lengths |
| Decoding | Time per output token, throughput and batch scaling |
| KV cache | Bytes per token, maximum batch size and eviction behaviour |
| Long context | Accuracy by evidence position and context length |
| Correctness | Causal and padding-mask tests against a trusted reference |

Claims of linear complexity or memory savings should state sequence length, hardware, precision, batch size and kernel. Asymptotic complexity alone does not predict wall-clock performance.

## Limitations and takeaways

Dense attention remains expensive for long sequences, and a KV cache shifts rather than eliminates inference memory costs. Sparse and approximate alternatives introduce kernel, quality and portability trade-offs. Longer context also does not guarantee reliable use of distant evidence.

The useful mental model is:

- attention performs content-dependent mixing of values;
- masks determine which query-key pairs are visible;
- positional mechanisms supply order and distance;
- self- and cross-attention describe where inputs originate;
- sparse, approximate and IO-efficient methods solve different efficiency problems;
- KV caching accelerates decoding but is not persistent memory; and
- attention weights require validation before becoming explanations.

## References

1. Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS*.
2. Beltagy, I., Peters, M. E. and Cohan, A. (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
3. Zaheer, M. et al. (2020). [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). *NeurIPS*.
4. Choromanski, K. et al. (2021). [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794). *ICLR*.
5. Dao, T. et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). *NeurIPS*.
6. Su, J. et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
7. Press, O., Smith, N. A. and Lewis, M. (2022). [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409). *ICLR*.
8. Shazeer, N. (2019). [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150).
9. Ainslie, J. et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). *EMNLP*.
10. Jain, S. and Wallace, B. C. (2019). [Attention is not Explanation](https://aclanthology.org/N19-1357/). *NAACL-HLT*.
11. Wiegreffe, S. and Pinter, Y. (2019). [Attention is not not Explanation](https://aclanthology.org/D19-1002/). *EMNLP-IJCNLP*.
