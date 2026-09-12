---
title: "LLM Agents: From Model Output to Reliable Action"
date: 2025-05-05T09:00:00+01:00
lastmod: 2026-09-13T01:10:00+02:00
draft: false
tags: ["llm", "agents", "tool-use", "evaluation", "ai-safety", "production-ai"]
weight: 115
math: true
showtoc: true
description: "A guide to reliable LLM agents: typed tools, permissions, retries, human approval, observability, security, and trajectory-level evaluation."
---

An LLM becomes an agent when its outputs can change an environment: querying a database, editing a record, sending a message, running code, or asking a person for approval. The model remains important, but the production system around it determines whether those actions are valid, authorised, repeatable, and observable.

This distinction matters. A capable model inside a weak control loop can execute the wrong tool, repeat an irreversible action after a timeout, or follow malicious instructions hidden in retrieved content. A dependable agent therefore needs more than prompting. It needs an explicit action interface, an execution policy, state management, safety boundaries, and evaluation over complete trajectories.

This article develops that system view. It focuses on tool-using agents that operate through APIs rather than agents embodied in physical environments.

## A useful definition

At time step \(t\), let the agent receive an observation \(o_t\), maintain state \(s_t\), and choose an action \(a_t\). Its history is

$$
h_t = (o_1, a_1, o_2, a_2, \ldots, o_t).
$$

The model defines a conditional distribution over candidate outputs:

$$
z_t \sim p_\theta(z \mid h_t, c_t),
$$

where \(c_t\) contains system instructions, tool schemas, permissions, and relevant memory. A decoder may sample, use constrained decoding, or select a high-probability result; it is not necessarily an \(\arg\max\). The output \(z_t\) is then parsed and validated into an action:

$$
a_t = V(D(z_t), P_t),
$$

where \(D\) decodes structured output, \(V\) validates it, and \(P_t\) is the policy and permission context. Invalid output produces a controlled error or repair attempt rather than a tool execution.

The environment—not the LLM—executes the action and returns an observation:

$$
o_{t+1}, s_{t+1} = E(s_t, a_t).
$$

This separation creates three useful trust boundaries:

1. The model proposes an action.
2. The control layer decides whether the action may run.
3. The tool or environment executes it and reports the result.

Calling the entire system an “LLM policy” hides these boundaries. In production, they are where most reliability and security work belongs.

## The controlled execution loop

A minimal agent loop should make every transition explicit:

```python
for step in range(max_steps):
    context = assemble_context(task, state, observations, tool_schemas)
    proposal = model.generate(context)

    action = parse_and_validate(proposal)
    authorise(action, user_permissions, policy)

    if action.requires_approval:
        action = await obtain_human_approval(action)

    result = execute_with_idempotency(action)
    record_trace(step, action, result)

    state = reduce_state(state, result)
    if result.is_terminal:
        return result

return fail("step budget exhausted")
```

Each function is a control point. `parse_and_validate` enforces the schema. `authorise` applies deterministic policy. `obtain_human_approval` gates consequential actions. `execute_with_idempotency` prevents duplicate side effects. `record_trace` makes the trajectory inspectable. The LLM should not be responsible for implementing these guarantees in natural language.

### Typed actions

Tools should expose narrow schemas rather than a general command channel. For example:

```json
{
  "name": "create_refund",
  "arguments": {
    "order_id": "ORD-1842",
    "amount_gbp": 24.50,
    "reason_code": "duplicate_charge"
  }
}
```

The runtime can validate types, required fields, bounds, enumerations, and resource ownership before execution. A schema prevents malformed calls; it does not prove that the proposed action is correct or permitted. Those checks belong to policy and application logic.

### State and memory

Working state includes the current goal, completed actions, tool results, unresolved questions, and remaining budget. It should be represented explicitly rather than reconstructed from an ever-growing transcript.

Longer-term memory is optional. When it is useful, separate at least three concerns:

- **Retrieval:** which past records are relevant?
- **Provenance:** where did each record originate, and when was it valid?
- **Write policy:** which observations are allowed to become persistent memory?

An embedding similarity score can help retrieve candidates,

$$
m^* = \arg\max_{m \in \mathcal{M}} \operatorname{sim}(q, e_m),
$$

but similarity is not authority or truth. Retrieved memory remains data that may be stale, conflicting, or adversarial.

## Reasoning, acting, and planning

[ReAct](https://arxiv.org/abs/2210.03629) showed that language models can interleave reasoning traces with actions and observations. In that formulation, the model produces reasoning and an action; the external environment supplies the observation. The observation must not be described as part of the model’s generated triple.

The important engineering idea is feedback: the system can inspect a tool result before choosing the next action. This can support plan updates and exception handling, but it does not guarantee that errors will decrease. A bad observation, incorrect state update, or persuasive prompt injection can instead compound failure.

Production systems also need not expose private chain-of-thought. A concise, auditable action rationale—goal, evidence used, constraint checked, and expected effect—is usually more useful than storing unrestricted hidden reasoning.

### Planning is not one data structure

A plan may be:

- a linear checklist for a stable workflow;
- a state machine for processes with known transitions;
- a dependency graph for tasks that can run in parallel;
- a search tree when alternatives require exploration;
- a policy that chooses the next action from the current state;
- or a classical planner operating over explicit preconditions and effects.

A directed acyclic graph is therefore one possible representation, not a definition of planning. Cycles are legitimate when the system must gather evidence, validate it, and revise the plan. The control layer should detect unproductive loops using repeated-state checks, step budgets, and progress criteria.

Plan quality also depends on the environment. When preconditions and effects can be formalised, a symbolic planner may provide stronger guarantees than free-form plan generation. [LLM+P](https://arxiv.org/abs/2304.11477), for example, combines an LLM interface with an external classical planner rather than asking the language model to solve the planning problem alone.

## Reliability is an execution property

Retries are necessary for transient failures, but a retry policy must understand side effects.

| Failure | Appropriate response |
|---|---|
| Rate limit or temporary service failure | Retry with bounded exponential backoff and jitter |
| Invalid tool arguments | Return a structured validation error; allow a limited repair attempt |
| Permission denied | Stop; do not ask the model to work around the policy |
| Ambiguous user intent | Ask the user or route to human review |
| Unknown outcome after a write timeout | Reconcile by idempotency key or read-after-write before retrying |
| Repeated state with no progress | Terminate or escalate |

### Idempotency and recovery

Suppose an agent submits a payment request and the connection times out. Repeating the call blindly can create two payments. Each logical operation should carry a stable idempotency key, and the executor should persist the action state:

$$
\text{proposed} \rightarrow \text{authorised} \rightarrow \text{running} \rightarrow \text{succeeded or failed}.
$$

On recovery, the system checks the recorded operation before deciding whether to retry. Where a tool cannot support idempotency, isolate it behind an adapter or require approval before repeating an uncertain action.

### Budgets and termination

Agents need limits on steps, wall-clock time, model tokens, tool calls, and monetary cost. A useful run-level cost model is descriptive rather than asymptotically vague:

$$
C_{\text{run}} = \sum_{t=1}^{T}
\left(C_{\text{model},t} + C_{\text{tool},t} + C_{\text{infra},t}\right).
$$

Latency is similarly the sum of model, tool, queueing, approval, and retry delays along the critical path. Parallel tool calls can reduce wall-clock time when they are independent, but they also increase concurrency, ordering, and cancellation complexity.

## Permissions and human approval

The model should receive only the tools and resources needed for the current task. Enforcement must occur outside the prompt.

A practical permission model distinguishes:

- read from write operations;
- reversible from irreversible actions;
- resource scope, such as one account or repository;
- value limits, such as a maximum refund amount;
- data classification and destination;
- whether human approval is required.

Approval should present the exact proposed action, its target, material parameters, evidence, and expected side effect. The approved payload should be bound to execution so the agent cannot change it afterwards. High-risk examples include sending external messages, publishing content, deleting records, moving money, changing access controls, or executing code outside a sandbox.

## Prompt injection is a trust-boundary problem

Tool outputs, retrieved documents, webpages, emails, and memory entries are untrusted data. They can contain text that attempts to override the agent’s instructions. Research on [indirect prompt injection](https://arxiv.org/abs/2302.12173) demonstrated that malicious instructions embedded in external content can manipulate tool-using LLM applications.

No single prompt reliably solves this problem. Defence should be layered:

1. Label and isolate untrusted content in the context.
2. Give the agent least-privilege credentials and a task-scoped tool set.
3. Validate every action against deterministic policy.
4. Prevent secrets from entering contexts or tools that do not need them.
5. Require approval for consequential or cross-boundary actions.
6. Constrain network destinations and data egress.
7. Test with adversarial content in documents and tool responses.

Input filtering can reduce obvious attacks, but it should not be the final security boundary. Assume that some malicious instructions will reach the model and ensure the executor still blocks unauthorised effects.

## Observability: trace the trajectory

A final answer is insufficient for diagnosing an agent. Record a structured event for each step:

- run, user, task, model, prompt, and policy versions;
- proposed action and validated arguments;
- permission and approval decisions;
- tool request, response status, latency, and retry count;
- state transition and termination reason;
- token use and estimated cost;
- provenance for retrieved evidence;
- redacted error details.

Logs must avoid secrets, sensitive document content, and unrestricted reasoning traces. Use identifiers or hashes where full payload retention is unnecessary, apply access controls, and set retention periods deliberately.

Distributed traces help separate model latency from tool latency and reveal loops, redundant retrieval, repeated calls, or approval bottlenecks. Replay is useful only when external dependencies and side effects are mocked or safely isolated.

## Evaluate outcomes and behaviour

Agent evaluation should cover the result, the path taken, and repeatability. [AgentBench](https://arxiv.org/abs/2308.03688) evaluates agents across interactive environments and identifies long-term reasoning, decision-making, and instruction following as recurring failure areas. [τ-bench](https://arxiv.org/abs/2406.12045) adds tool-agent-user interaction and evaluates whether repeated trials reliably satisfy the task and policy.

A production test set should contain normal tasks, edge cases, policy conflicts, tool failures, ambiguous requests, and adversarial observations. For each run, score dimensions such as:

| Dimension | Example measure |
|---|---|
| Task outcome | Goal-state match or verified task success |
| Policy compliance | Forbidden-action rate; required approvals obtained |
| Tool correctness | Valid call rate; argument and resource accuracy |
| Trajectory quality | Unnecessary calls, loops, invalid transitions, recovery quality |
| Reliability | Success across repeated runs, seeds, or model samples |
| Efficiency | Tokens, calls, cost, and latency per successful task |
| Human burden | Approval and correction rate |
| Security | Attack success rate on injection and exfiltration tests |

Pass rate from one attempt hides nondeterminism. If \(X_i\) indicates success on trial \(i\), estimate

$$
\hat{p} = \frac{1}{n}\sum_{i=1}^{n} X_i
$$

and report uncertainty or repeated-run criteria. For critical workflows, measure the probability of consistent success across several trials, not only whether one run happened to pass.

Trajectory assertions catch failures that the final state misses. An agent might reach the correct outcome after accessing an unauthorised record, making a redundant write, or violating an approval rule. Evaluation should therefore inspect state transitions and side effects as first-class outputs.

## When not to build an agent

Use a deterministic workflow when the process is known, the inputs are structured, and each transition can be specified in code. Add an LLM only where language understanding or flexible judgement is genuinely required. Use retrieval when the task is primarily finding and synthesising information. Use a single tool call when one call is enough.

An agent is justified when the next action depends on intermediate observations and the path cannot be enumerated economically in advance. Even then, keep deterministic orchestration around probabilistic decisions. The most reliable design often gives the model a small, bounded choice inside a larger conventional system.

## Practical design checklist

Before deploying a tool-using agent, verify that:

- actions use typed, versioned schemas;
- authentication and authorisation are enforced outside the model;
- tool access follows least privilege;
- consequential actions require bound human approval;
- writes use idempotency keys and recoverable state transitions;
- retries are bounded and distinguish transient from permanent failures;
- untrusted content cannot directly grant instructions or permissions;
- step, time, token, tool, and cost budgets terminate runaway loops;
- traces expose decisions, calls, errors, latency, and cost without leaking sensitive data;
- evaluation covers outcomes, trajectories, policy compliance, attacks, and repeated runs;
- deterministic workflows remain deterministic where possible.

## Takeaway

An LLM agent is not simply a model that “reasons and acts”. It is a controlled software system in which a probabilistic model proposes actions and deterministic components constrain, execute, observe, and audit them.

The design question is therefore not only whether the model can complete a task. It is whether the whole system can complete it repeatedly, within permissions and budget, while producing enough evidence to understand every consequential side effect.

## References

1. Yao, S. et al. (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). arXiv:2210.03629.
2. Liu, B. et al. (2023). [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477). arXiv:2304.11477.
3. Liu, X. et al. (2023). [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688). arXiv:2308.03688.
4. Yao, S. et al. (2024). [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045). arXiv:2406.12045.
5. Greshake, K. et al. (2023). [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173). arXiv:2302.12173.
