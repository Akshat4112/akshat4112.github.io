---
title: "LLM Agents: Building AI Systems That Can Reason and Act"
date: 2025-05-05T09:00:00+01:00
draft: false
tags: ["llm", "agents", "deep-learning", "ai", "reinforcement-learning", "decision-making", "autonomous-systems"]
weight: 115
math: true
author: "Akshat Gupta"
showtoc: true
description: "A theory and mathematics-first guide to LLM agents — how autonomous AI systems reason, plan, act, and remember, with formal definitions and architectural principles."
---

Large Language Models (LLMs), like [GPT-3](https://arxiv.org/abs/2005.14165), GPT-4, and others, have taken the world by storm due to their impressive language generation and understanding capabilities. However, when these models are augmented with decision-making capabilities, memory, and actions in specific environments, they become something fundamentally more powerful.

Enter **LLM Agents** — autonomous systems built on top of large language models that can pursue goals, use tools, plan multi-step actions, and adapt based on feedback.

---

## 🧠 What Is an LLM Agent?

Formally, an agent is a function that maps a history of observations to an action:

$$\pi: \mathcal{H} \rightarrow \mathcal{A}$$

where $\mathcal{H} = (o_1, a_1, o_2, a_2, \ldots, o_t)$ is the trajectory of observations and actions so far, and $\mathcal{A}$ is the set of possible actions.

In an LLM agent, this policy $\pi$ is implemented by the language model itself. Given a context (the history), the LLM generates the next action token-by-token via:

$$a_t = \arg\max_{a} P_{\theta}(a \mid \mathcal{H}_t)$$

where $\theta$ are the model's parameters and $P_\theta$ is the language model's distribution over possible next outputs.

This is the core insight: **an LLM's text generation is itself a policy over actions**.

---

## 🛠️ Core Components of an LLM Agent

A fully-featured LLM agent has four primary components:

### 1. Foundation Model (The Core)

The language model $\mathcal{M}_\theta$ provides reasoning, planning, and generation. It serves as the "brain" — all decision-making flows through it. Models like GPT-4, Claude, or [Llama 2](https://github.com/meta-llama/llama) are typical choices.

The model takes as input a structured prompt containing the task, available tools, memory, and current state, and generates structured output specifying the next action.

### 2. Tool Use

An agent has access to a set of tools $\mathcal{T} = \{t_1, t_2, \ldots, t_k\}$, where each tool $t_i$ is a function:

$$t_i: \mathcal{X}_i \rightarrow \mathcal{Y}_i$$

mapping an input (e.g., a search query) to an output (e.g., a list of results). The agent must:

1. **Select** which tool to use: $t^* = \text{select}(a_t, \mathcal{T})$
2. **Parameterise** the tool call: $x^* = \text{parse}(a_t)$
3. **Execute**: $y = t^*(x^*)$
4. **Incorporate** the result back into the context: $\mathcal{H}_{t+1} \leftarrow \mathcal{H}_t \cup \{y\}$

### 3. Memory

Agent memory can be categorised by time scale and mechanism:

| Memory Type | Scope | Mechanism |
|---|---|---|
| **Working memory** | Current session | LLM context window |
| **Episodic memory** | Past interactions | External key-value store |
| **Semantic memory** | World knowledge | Vector database (RAG) |
| **Procedural memory** | How to act | Fine-tuned weights |

Retrieval from episodic or semantic memory is typically done via nearest-neighbour search:

$$m^* = \arg\max_{m \in \mathcal{M}} \text{sim}(\mathbf{q}, \mathbf{e}_m)$$

where $\mathbf{q}$ is the query embedding, $\mathbf{e}_m$ is the embedding of memory entry $m$, and $\text{sim}$ is cosine similarity.

### 4. Planning and Reasoning

An agent decomposes a goal $G$ into a sequence of sub-goals:

$$G \rightarrow (g_1, g_2, \ldots, g_n)$$

This can be modelled as a search problem over a directed acyclic graph (DAG), where:
- **Nodes** are states (current progress toward $G$)
- **Edges** are actions $a \in \mathcal{A}$
- **Leaf nodes** are terminal states (success or failure)

The agent seeks a path from the initial state $s_0$ to a goal state $s^*$ that minimises cost (number of steps, resource usage, etc.).

---

## 📐 The ReAct Framework

The [ReAct](https://arxiv.org/abs/2210.03629) framework (Yao et al., 2022) is the most widely adopted paradigm for structuring LLM agent behaviour. It interleaves **reasoning** and **acting** in a single generation loop.

Formally, at each step $t$ the agent produces a triple:

$$(\tau_t, a_t, o_t)$$

where:
- $\tau_t$ = **Thought** — a natural language reasoning trace
- $a_t$ = **Action** — a tool call or final answer
- $o_t$ = **Observation** — the result of executing $a_t$

The loop continues until the agent emits a terminal action (FINISH), or a maximum step count $T$ is reached:

$$\mathcal{H}_T = (\tau_1, a_1, o_1, \tau_2, a_2, o_2, \ldots, \tau_T, a_T, o_T)$$

The key insight is that interleaving reasoning ($\tau_t$) with action ($a_t$) dramatically reduces compounding errors — the model can "course-correct" at each step rather than committing to a full plan upfront.

---

## 🏗️ Advanced Architectures

### Multi-Agent Systems

In multi-agent systems, a set of $N$ specialised agents $\{\mathcal{A}_1, \ldots, \mathcal{A}_N\}$ collaborate on a shared task. Each agent $\mathcal{A}_i$ has a role $r_i$ (e.g., Researcher, Critic, Synthesizer) and its own policy $\pi_i$.

The system can be modelled as a **communication graph** $\mathcal{G} = (V, E)$ where nodes are agents and directed edges $(i, j) \in E$ represent information flow from agent $i$ to agent $j$.

Common topologies:
- **Pipeline**: $\mathcal{A}_1 \rightarrow \mathcal{A}_2 \rightarrow \cdots \rightarrow \mathcal{A}_N$ — sequential handoff
- **Debate**: all agents communicate with all others; a mediator synthesises
- **Hierarchical**: a coordinator agent manages sub-agents, delegating sub-goals

### Hierarchical Planning

For complex goals, a hierarchical planner introduces levels of abstraction. At level $l$, the agent operates on a coarser action space $\mathcal{A}^{(l)}$:

$$\mathcal{A}^{(0)} \subset \mathcal{A}^{(1)} \subset \cdots \subset \mathcal{A}^{(L)}$$

A high-level plan is generated at level $L$, then each step is recursively decomposed until primitive (executable) actions at level $0$ are reached. This mirrors hierarchical reinforcement learning (HRL) approaches like the options framework.

### Reflexion: Self-Evaluation and Improvement

[Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) enables agents to improve across trials via verbal self-reflection. After each attempt $k$, a feedback model $\mathcal{F}$ scores the trajectory:

$$s_k = \mathcal{F}(\mathcal{H}_k, G)$$

The agent generates a verbal reflection $r_k$ summarising what went wrong and stores it in a **reflection buffer** $\mathcal{B}$. On attempt $k+1$, the buffer is prepended to the context:

$$\mathcal{H}_0^{(k+1)} \leftarrow \mathcal{B}_k \cup \{G\}$$

This allows the agent to condition on past mistakes without gradient updates — an in-context analogue of reinforcement learning.

---

## 📊 Evaluating LLM Agents

Evaluating agents is substantially harder than evaluating static models. Key dimensions:

| Metric | Definition |
|---|---|
| **Success Rate** | $\frac{\text{tasks completed}}{\text{tasks attempted}}$ |
| **Efficiency** | Average steps $\mathbb{E}[T]$ to task completion |
| **Autonomy** | Fraction of steps requiring no human correction |
| **Robustness** | Success rate under distributional shift |

Standard benchmarks include **WebShop** (instruction-following in simulated e-commerce), **ALFWorld** (text-based embodied navigation), and **HotPotQA** (multi-hop question answering requiring tool use).

---

## 🚀 Applications

| Domain | Agent Role |
|---|---|
| Research | Literature search, hypothesis generation, experiment design |
| Software Engineering | Code generation, test writing, debugging |
| Business Automation | Document processing, workflow orchestration, report generation |
| Healthcare | Medical record analysis, differential diagnosis support |
| Education | Personalised tutoring, adaptive content generation |
| Customer Support | Intent classification, knowledge retrieval, escalation routing |

---

## 🧪 Open Challenges

**1. Long-Horizon Planning**
Current agents struggle when tasks require $T \gg 10$ steps. Errors compound: a mistake at step $t$ propagates through all subsequent observations $o_{t+1}, \ldots, o_T$.

**2. Grounding and Hallucination**
The policy $\pi_\theta$ can generate plausible-sounding but factually incorrect actions, especially when the required knowledge is absent from the context or training data.

**3. Safety and Alignment**
Autonomous action in the real world raises the question: how do we constrain $\mathcal{A}$ to exclude harmful actions? Current approaches include prompt-based guardrails, constitutional AI, and human-in-the-loop approval for irreversible actions.

**4. Efficiency**
Each reasoning step requires a full forward pass through the LLM. For a $T$-step trajectory, the total compute scales as $O(T \cdot |\mathcal{H}_T| \cdot d_{\text{model}})$, which becomes prohibitive for long tasks with large models.

**5. Evaluation**
Unlike static benchmarks, agent tasks are non-deterministic and path-dependent. Two agents can both succeed but with very different strategies — and one may generalise far better than the other.

---

## 🧠 Final Thoughts

LLM agents represent a fundamental shift: from models that *respond* to models that *pursue goals*. The theoretical machinery — policies over action spaces, tool-augmented generation, memory retrieval, hierarchical planning, and self-reflection — provides a rigorous foundation for understanding why these systems work and where they fail.

The frontier is moving fast. Multi-agent collaboration, persistent memory, and tighter integration with formal reasoning systems are likely to define the next generation of agent architectures.

— **Akshat**
