---
title: "Model Extraction Attacks: How Hackers Steal AI Models"
date: 2024-09-15T09:00:00+01:00
draft: false
tags: ["ai-security", "model-extraction", "machine-learning", "cybersecurity", "llm", "deep-learning", "ai"]
weight: 113
math: true
showtoc: true
description: "A theory and math-first explainer on model extraction attacks — how adversaries steal ML models through queries, the formal threat model, and principled defences."
---

Training a state-of-the-art machine learning model is expensive. Large language models like [GPT-3](https://arxiv.org/abs/2005.14165) required hundreds of petaflop-days of compute and millions of dollars. Yet once deployed behind an API, they are vulnerable to a surprisingly subtle attack: an adversary who never sees the weights, never reads the training data, and never touches the server — but can still *steal the model* by asking it questions.

This is a **model extraction attack**, and it is one of the more underappreciated threats in production ML security. Related adversarial work — see [Goodfellow et al.](https://arxiv.org/abs/1412.6572) on FGSM — focuses on perturbing inputs to fool a model. Model extraction goes further: the attacker wants a *copy* of the model itself.

---

## 🎯 Formal Threat Model

Let the target model be a function:

$$f^*: \mathcal{X} \rightarrow \mathcal{Y}$$

mapping inputs $x \in \mathcal{X}$ to outputs $y \in \mathcal{Y}$ (e.g., class probabilities, logits, or generated text). The attacker has no access to $f^*$'s parameters $\theta^*$ or training data $\mathcal{D}$. Their only interface is an oracle:

$$\mathcal{O}(x) = f^*(x)$$

The attacker's goal is to construct a surrogate model $\hat{f}$ that approximates $f^*$:

$$\hat{f} \approx f^* \quad \text{such that} \quad \mathbb{E}_{x \sim \mathcal{P}}[\ell(\hat{f}(x), f^*(x))] < \varepsilon$$

for some loss $\ell$ and input distribution $\mathcal{P}$, with as few oracle queries as possible.

---

## 🗂️ Attack Taxonomy

### Black-Box Attacks

The adversary can only observe inputs and outputs — no knowledge of architecture, weights, or training procedure. This is the most realistic and most studied setting.

The attacker constructs a **query dataset** $\mathcal{D}_q = \{(x_i, \mathcal{O}(x_i))\}_{i=1}^n$ by selecting inputs $x_i$ and recording the oracle's responses, then trains $\hat{f}$ on $\mathcal{D}_q$:

$$\hat{\theta} = \arg\min_\theta \sum_{i=1}^n \ell(\hat{f}_\theta(x_i), \mathcal{O}(x_i))$$

The key question is *how to choose* $x_i$. A random distribution wastes queries on uninformative regions of $\mathcal{X}$; adaptive strategies focus queries where the oracle's decision boundary is most informative.

### White-Box Attacks

The adversary has full access to the model — its architecture, weights, and sometimes training data. This typically arises from a misconfigured API, leaked model file, or insider access. White-box extraction is trivially easy (copy the weights directly) and is primarily a deployment security problem rather than a learning theory one.

The interesting problem — and the focus of research — is black-box extraction.

---

## 📐 The Mathematics of Extraction

### Model Distillation as Extraction

Model extraction is structurally identical to **knowledge distillation** (Hinton et al., 2015), except that distillation is a cooperative compression technique and extraction is adversarial.

In distillation, a student $\hat{f}$ is trained to match the soft output distribution of a teacher $f^*$. The loss uses the **Kullback-Leibler divergence**:

$$\mathcal{L}_\text{KD} = \tau^2 \cdot D_\text{KL}(\sigma(f^*(x)/\tau) \| \sigma(\hat{f}(x)/\tau))$$

where $\sigma$ is the softmax function and $\tau > 1$ is a temperature that softens the probability distribution, preserving more information about the teacher's beliefs across all classes — not just the top prediction.

Soft labels are far more informative than hard labels. If $f^*$ outputs $[0.7, 0.28, 0.02]$ rather than just class 0, the attacker learns both the decision boundary *and* how confident the model is near it.

### Query Complexity

How many queries $n$ does an attacker need? This depends on the complexity of $f^*$. For a linear classifier in $\mathbb{R}^d$, the decision boundary has $d$ degrees of freedom, so $O(d)$ queries suffice. For a neural network with $p$ parameters, extraction in the worst case requires $O(p)$ queries — but in practice far fewer, because real-world models have low intrinsic dimensionality in their decision surfaces.

Formally, the extraction error is bounded by a generalisation-style argument:

$$\mathbb{E}[\ell(\hat{f}, f^*)] \leq \mathcal{O}\!\left(\sqrt{\frac{\text{VC}(\hat{f})}{n}}\right)$$

where $\text{VC}(\hat{f})$ is the VC dimension of the surrogate model class and $n$ is the number of queries. Increasing $n$ or restricting the expressiveness of the surrogate class reduces extraction error.

### Active Query Selection

Passive (random) querying is inefficient. Active strategies select the most informative $x_i$ at each step. A natural criterion is **uncertainty sampling**: query the oracle at points where the current surrogate is most uncertain:

$$x_i^* = \arg\max_{x} H(\hat{f}(x))$$

where $H$ is the entropy of the surrogate's output distribution. High entropy means the surrogate doesn't yet know what to predict there — so the oracle's response is maximally informative.

This turns extraction into an active learning problem, substantially reducing the query budget.

---

## ⚠️ Why Soft Outputs Are the Attacker's Best Friend

Most APIs return not just the predicted class but a full probability vector (confidence scores). This is a serious vulnerability.

Consider a binary classifier. A hard-label response tells the attacker which side of the boundary $x$ falls on — one bit of information. A soft-label response like $f^*(x) = 0.83$ also reveals *how far* from the boundary $x$ is. Aggregating many such responses allows the attacker to reconstruct the shape of the decision boundary with far fewer queries than hard labels would require.

**The practical implication**: APIs that return only the top-1 prediction (hard labels) are significantly harder to extract from than those that return full probability distributions.

---

## 🛡️ Defences

### 1. Query Rate Limiting

The simplest defence is to bound the number of queries per user over a time window. If the attacker needs $n_\text{extract}$ queries to achieve extraction error $\varepsilon$, rate limiting forces the attack to take time $\geq n_\text{extract} / r$ for rate $r$ — raising the cost.

### 2. Output Perturbation

Adding noise $\eta$ to the model's output before returning it:

$$\tilde{f}(x) = f^*(x) + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

increases the attacker's effective loss floor. The surrogate trained on noisy labels has a floor on how well it can approximate $f^*$:

$$\mathbb{E}[\ell(\hat{f}, f^*)] \geq \Omega(\sigma^2)$$

The tradeoff: too much noise degrades the API's usefulness for legitimate users.

### 3. Model Watermarking

A watermark embeds a set of **backdoor trigger-response pairs** $\{(x_w^{(i)}, y_w^{(i)})\}$ into $f^*$ during training. These are inputs that produce a specific, unusual output that no naturally-trained model would produce.

If an extracted model $\hat{f}$ is later suspected, the owner queries it on $\{x_w^{(i)}\}$ and checks whether it reproduces $\{y_w^{(i)}\}$. Probability of a false positive is:

$$\Pr[\hat{f}(x_w^{(i)}) = y_w^{(i)} \text{ by chance}] = \prod_{i=1}^k \frac{1}{|\mathcal{Y}|}$$

For $k = 10$ watermark queries on a 10-class classifier, this is $10^{-10}$ — essentially zero. Watermarking provides **cryptographic-strength provenance** for model ownership.

### 4. Differential Privacy

[Differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) during training bounds how much any single training point influences the model's outputs. A model trained with $(\varepsilon, \delta)$-DP satisfies:

$$\Pr[f^*(x) \in S] \leq e^\varepsilon \cdot \Pr[f^{*\prime}(x) \in S] + \delta$$

for any neighbouring datasets differing by one example. This limits the information an attacker can extract about training data via the oracle, reducing the attack's utility for membership inference and related attacks.

---

## 🔬 My Research Context

At Validaitor, I built **model security assessment workflows** including copycat and model-stealing attack simulations. This involved evaluating how many queries a surrogate needed to reach target fidelity across different model types (classifiers, embedding models, LLMs), and implementing watermarking and fingerprinting as countermeasures. The most practically effective defence was a combination of output truncation (returning only top-3 probabilities) and rate limiting — neither alone was sufficient for a determined adversary.

---

## 🧠 Final Thoughts

Model extraction is a clean, well-posed problem that sits at the intersection of learning theory, information theory, and adversarial ML. The attacker and defender are playing a minimax game:

$$\min_{\hat{f}} \max_{\text{query strategy}} \;\mathbb{E}[\ell(\hat{f}, f^*)]$$

subject to the oracle budget constraint. Defences that raise the query cost, degrade the informativeness of outputs, or embed verifiable watermarks each attack a different term in this objective.

As ML APIs become more powerful and more profitable, model extraction will only grow in importance. The field is moving toward **certified defences** — provable bounds on extraction fidelity rather than empirical evaluations against known attack strategies.

— Akshat
