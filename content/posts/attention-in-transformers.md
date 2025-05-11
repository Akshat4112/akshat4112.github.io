---
title: "Understanding Attention in Transformers: The Core of Modern NLP"
date: 2024-08-15T09:00:00+01:00
draft: false
tags: ["transformers", "attention", "deep-learning", "NLP", "self-attention", "neural-networks", "AI"]
weight: 112
math: true
---
+++
title = 'Understanding Attention in Transformers: The Core of Modern NLP'
date = 2025-05-11T22:15:00+02:00
draft = false
tags = ["transformers", "attention", "deep-learning", "NLP", "math"]
weight = 109
+++

When people say “Transformers revolutionized NLP,” what they *really* mean is:
> **Attention** revolutionized NLP.

From GPT and BERT to LLaMA and Claude, **attention mechanisms** are the beating heart of modern large language models.

But what exactly is attention? Why is it so powerful? And how many types are there?

Let’s dive in.

---

## 🧠 What is Attention?

In the simplest sense, **attention is a way for a model to focus on the most relevant parts of the input when generating output**.

It answers:
> “Given this word, which other words should I pay attention to — and how much?”

---

## 🔢 The Scaled Dot-Product Attention

Let’s break it down mathematically.

Given:
- Query matrix \( Q \in \mathbb{R}^{n \times d_k} \)
- Key matrix \( K \in \mathbb{R}^{n \times d_k} \)
- Value matrix \( V \in \mathbb{R}^{n \times d_v} \)

The attention output is:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

- \( QK^\top \): Dot product measures similarity
- \( \sqrt{d_k} \): Scaling factor to avoid large softmax values
- softmax: Turns similarity into attention weights
- \( V \): Weighted sum of value vectors

📖 Citation: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) (Attention is All You Need)

---

## 🔁 Multi-Head Attention

Instead of applying one attention function, we apply it **multiple times in parallel** with different projections.

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
\]
\[
\text{where head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
\]

Each head learns different types of relationships (e.g., syntactic, semantic).

---

## 🧩 Types of Attention in Transformers

Let’s look at the key attention variations used in different transformer architectures.

### 1. **Self-Attention**
- Query, Key, and Value come from the **same input**.
- Used in encoder and decoder blocks.
- Each token attends to **every other token** (or just previous ones in causal attention).

\[
\text{SelfAttention}(X) = \text{Attention}(X, X, X)
\]

### 2. **Cross-Attention**
- Used in encoder-decoder models like T5 or BART.
- Query comes from decoder, Key & Value come from encoder output.

\[
\text{CrossAttention}(Q_{\text{decoder}}, K_{\text{encoder}}, V_{\text{encoder}})
\]

### 3. **Masked (Causal) Attention**
- Used in autoregressive models like GPT.
- Prevents tokens from attending to future tokens.
- Enforced using a **triangular mask**.

\[
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
\]
where \( M \) is a mask with \( -\infty \) in upper triangle.

### 4. **Local Attention**
- Each token only attends to a **local window** (e.g., ±128 tokens).
- Reduces compute from \( O(n^2) \) to \( O(nw) \)

Used in models like Longformer ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150))

### 5. **Sparse Attention**
- Instead of full attention, use pre-defined patterns (e.g., strided, global).
- Reduces memory usage.

Examples:
- **BigBird** ([Zaheer et al., 2020](https://arxiv.org/abs/2007.14062))
- **Reformer** ([Kitaev et al., 2020](https://arxiv.org/abs/2001.04451))

### 6. **Linear / Kernelized Attention**
- Approximates attention with linear complexity.
- Replace softmax with kernel function:
\[
\text{Attention}(Q, K, V) = \phi(Q)(\phi(K)^\top V)
\]

Used in **Performer** ([Choromanski et al., 2020](https://arxiv.org/abs/2009.14794))

### 7. **Memory-Augmented Attention**
- Adds **external memory vectors** (e.g., key-value cache, documents).
- Popular in **RAG** and **MoE** systems.

---

## 🏗️ Attention Block in Transformers

Each Transformer layer consists of:

1. **Multi-head Attention**
2. **Add & Layer Norm**
3. **Feed Forward Network**
4. **Add & Layer Norm**

```text
Input → [Multi-head Attention] → Add & Norm → [FFN] → Add & Norm → Output
