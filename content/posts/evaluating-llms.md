---
title: "How Do You Evaluate LLM Systems?"
date: 2024-06-15T09:00:00+01:00
lastmod: 2026-09-13T01:10:00+02:00
draft: false
tags: ["llm", "evaluation", "rag", "agents", "production-ai"]
weight: 110
math: true
showtoc: true
description: "A production-oriented framework for evaluating LLM applications across model quality, retrieval, agent behaviour, safety, latency, and cost."
---

A public benchmark can help compare foundation models. It cannot tell you whether a document assistant retrieves the right policy, whether an agent calls a tool with safe arguments, or whether a release stays within its latency budget.

For an applied AI system, evaluation is a decision process:

> Given a defined workload and risk tolerance, is this version safe and useful enough to release?

That question changes the unit of evaluation. The object under test is not only the model. It is the complete system: prompts, retrieval, tools, orchestration, guardrails, model configuration, and user interface.

This article develops a practical evaluation workflow for document-heavy and agentic applications. It covers dataset design, component metrics, human and model-based grading, failure analysis, uncertainty, and deployment gates.

## Start with the product decision

Before choosing a metric, write down the decision the evaluation must support. Typical decisions include:

- selecting a model for a fixed workflow;
- changing a prompt, retriever, or reranker;
- deciding whether a new agent policy can be released;
- checking whether a cost optimisation causes unacceptable regressions; or
- monitoring whether production behaviour has drifted.

Each decision needs an explicit evaluation contract:

| Contract field | Example for a policy assistant |
|---|---|
| Population | English questions from claims handlers |
| Required behaviour | Answer from approved policy documents and cite supporting passages |
| Critical failures | Unsupported answer, wrong policy version, disclosure of restricted content |
| Quality target | At least 95% citation correctness on critical cases |
| Operational target | p95 latency below 5 seconds and mean cost below £0.03 per request |
| Comparison rule | No critical-slice regression; statistically credible overall improvement |

This prevents a common failure: optimising an easy aggregate score while the behaviour that matters to users gets worse.

## Build a representative evaluation set

An evaluation set should model the workload, not merely provide a convenient collection of questions. I organise examples into four groups.

1. **Typical cases** represent the most frequent tasks and document types.
2. **Boundary cases** contain long documents, ambiguous requests, conflicting evidence, tables, OCR noise, or unusual tool outputs.
3. **Critical cases** exercise behaviour whose failure has a high consequence, such as access control, refusal, calculations, or policy eligibility.
4. **Known regressions** preserve production incidents and bugs as permanent tests.

Each example should carry metadata that supports slicing: task, language, document type, difficulty, source quality, user group, risk level, and expected behaviour. A single overall average can otherwise hide a severe regression in a small but important group.

Keep three datasets with different purposes:

- a **development set** for prompt and pipeline iteration;
- a **release set** that is not used during routine tuning; and
- a **production sample** that is periodically reviewed for drift and new failure modes.

Public benchmarks remain useful for broad capability checks, but they can be affected by test-set contamination and may not resemble the application workload ([Sainz et al., 2023](https://arxiv.org/abs/2310.18018); [Deng et al., 2023](https://arxiv.org/abs/2311.09783)). Treat them as supporting evidence rather than the release criterion.

## Decompose the system before scoring it

When a generated answer is wrong, a single end-to-end score does not identify the cause. The failure could be retrieval, generation, tool execution, orchestration, or presentation. Evaluate each boundary separately.

### Model and generation layer

Use deterministic checks wherever the task has an objectively verifiable result:

- exact match or token-level F1 for short answers;
- schema validity and field-level precision, recall, and F1 for extraction;
- executable unit tests for code;
- numerical tolerance for calculations; and
- citation entailment for evidence-backed answers.

Free-form quality needs a rubric. Define observable criteria such as correctness, completeness, relevance, instruction following, and calibrated abstention. Avoid one vague “quality” score: two responses with the same total can fail for very different reasons.

### Retrieval layer

For retrieval-augmented generation (RAG), evaluate retrieval independently from the answer. Useful metrics include:

\[
\text{Recall@}k = \frac{\text{relevant items retrieved in top }k}{\text{all relevant items}}
\]

and

\[
\text{Precision@}k = \frac{\text{relevant items retrieved in top }k}{k}.
\]

Also measure whether the required evidence appears at all, where it ranks, and how much irrelevant context is supplied. Then evaluate the generator for:

- **faithfulness**: are answer claims supported by the retrieved context?
- **answer correctness**: does the answer satisfy the reference or rubric?
- **citation correctness**: do citations point to passages that support the associated claims?
- **abstention quality**: does the system decline when evidence is missing or contradictory?

RAGAS formalised several reference-free metrics around retrieval relevance, faithfulness, and answer quality ([Es et al., 2023](https://arxiv.org/abs/2309.15217)). Such metrics can accelerate iteration, but they should be calibrated for the domain rather than accepted as ground truth.

### Agent layer

An agent can reach a correct final answer through an unsafe, expensive, or irreproducible trajectory. Record and score the complete trace:

- tool selection;
- argument validity;
- action order;
- state transitions;
- policy and permission compliance;
- recovery after tool failure;
- number of model and tool calls; and
- final task completion.

AgentBench demonstrates why interactive environments can reveal reasoning and decision-making failures that static prompts miss ([Liu et al., 2023](https://arxiv.org/abs/2308.03688)). For a production agent, add invariants that are specific to the application. For example: a write action must never occur before confirmation, restricted documents must not enter the model context, and a failed tool call must not be reported as success.

## Use a hierarchy of evaluators

No evaluator is sufficient for every output. Use the cheapest reliable method for each criterion.

### 1. Deterministic checks

Code should verify schemas, calculations, citations, permissions, tool arguments, latency, and cost. These checks are reproducible and should form the base of the suite.

### 2. Human review

Human reviewers are necessary for ambiguous, high-risk, or genuinely subjective criteria. Give reviewers a written rubric, examples of each score, and an explicit option for “cannot determine”. Measure agreement rather than assuming that one annotation is correct.

For categorical labels, Cohen's kappa for two raters is:

\[
\kappa = \frac{p_o - p_e}{1-p_e},
\]

where \(p_o\) is observed agreement and \(p_e\) is agreement expected by chance. Low agreement often means the rubric or task is underspecified, not that the reviewers are careless.

### 3. Model-based judges

An LLM judge is useful for high-volume, rubric-based screening. It is not an independent source of truth. The MT-Bench study found strong agreement with human preferences in its setting, while also documenting position, verbosity, self-enhancement, and reasoning biases ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)).

Before using a judge in a release gate:

1. Create a human-labelled calibration set containing both ordinary and difficult cases.
2. Freeze the judge model, prompt, rubric, decoding settings, and output schema.
3. Measure judge–human agreement for every important slice.
4. Inspect disagreements and revise the rubric before increasing automation.
5. Recalibrate after changing the judge, prompt, domain, or response distribution.

For pairwise grading, swap response order and require the decision to remain stable. Mask model identity. Ask the judge to cite evidence for factual criteria, and route uncertain or high-risk cases to people.

## A worked evaluation design

Consider a document assistant that answers questions from policy manuals and may call a calculator.

Build 500 release examples, stratified as follows:

| Slice | Examples | Primary checks |
|---|---:|---|
| Routine factual questions | 200 | Answer correctness, citation correctness, latency |
| Multi-document synthesis | 100 | Evidence coverage, faithfulness, completeness |
| Tables and OCR noise | 75 | Retrieval recall, numerical correctness |
| Missing or conflicting evidence | 75 | Correct abstention, uncertainty communication |
| Restricted or adversarial requests | 50 | Access control, refusal, data leakage |

For each system version, store the input, retrieved passages, prompt and configuration identifiers, raw model response, tool trace, token use, latency, and evaluator outputs. Replaying the same examples without versioned artefacts makes results difficult to explain or reproduce.

The evaluation pipeline can then apply:

```text
for example in release_set:
    trace = run_system(example, fixed_configuration)
    deterministic = run_contract_checks(trace, example)
    rubric_scores = judge(trace, example.rubric)
    record(example.metadata, trace, deterministic, rubric_scores)

compare(candidate, baseline, by=[task, risk, document_type])
send_disagreements_and_critical_failures_to_human_review()
apply_release_gates()
```

The important feature is not the particular sample count. It is the connection between realistic slices, observable failure modes, and a release decision.

## Compare systems with uncertainty

LLM outputs vary with sampling, infrastructure, and external tools. Run repeated trials for stochastic workflows when variance could change the decision. Compare a candidate and baseline on the same examples so that the analysis is paired.

Report both the effect size and uncertainty. For an accuracy-like metric on \(n\) examples, a rough standard error is:

\[
SE(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}.
\]

For complex metrics, use a paired bootstrap over examples to estimate a confidence interval for the difference. Do not declare a win from a tiny average increase if the interval includes a material regression.

Analyse the distribution as well as the mean:

- pass rate and critical-failure count;
- score by task and risk slice;
- p50, p95, and p99 latency;
- mean and tail token usage;
- cost per successful task; and
- variance across repeated runs.

HELM's multi-metric approach is a useful reminder that accuracy, robustness, fairness, toxicity, calibration, and efficiency can move in different directions ([Liang et al., 2022](https://arxiv.org/abs/2211.09110)).

## Turn failure analysis into new tests

Evaluation should produce a failure taxonomy, not only a dashboard. Review low scores and disagreements, then assign each failure to the earliest responsible component.

| Failure | Likely component | Follow-up test |
|---|---|---|
| Required passage absent | Retriever | Recall@k by document type and OCR quality |
| Passage present but claim unsupported | Generator | Claim-level faithfulness check |
| Correct tool, invalid arguments | Agent policy | Schema and boundary-value tests |
| Correct answer after excessive calls | Orchestration | Call-count and cost budget |
| Confident answer with no evidence | Product policy | Abstention and confidence rubric |

Add confirmed production failures to the regression set. Otherwise the evaluation suite remains static while the product and its users change.

## Define deployment gates

A release gate should combine non-negotiable constraints with comparative criteria. For example:

```text
Release the candidate only if:
- zero access-control or restricted-data failures occur;
- critical-slice citation correctness is at least 95%;
- no task slice regresses by more than 2 percentage points;
- the paired quality improvement is credible under the chosen interval;
- p95 latency remains below 5 seconds; and
- cost per successful task remains below £0.03.
```

The thresholds above are illustrative, not universal. Set them from consequence, user expectations, and operational constraints. In high-risk workflows, a small critical slice may deserve a stricter gate than a much larger collection of routine questions.

Offline evaluation is necessary but incomplete. After release, monitor the same quality and operational signals, sample real interactions with appropriate privacy controls, and maintain a rollback criterion. Online experiments can measure user behaviour, but they should not expose users to variants that already fail safety or correctness gates.

## What this framework does not solve

An evaluation suite is a model of reality, and every model has gaps.

- Human labels may encode ambiguous instructions or institutional bias.
- Model judges can fail systematically and silently.
- Production traffic can drift away from the release dataset.
- Rare harms may not appear in a feasible sample.
- Aggregate metrics can conceal subgroup failures.
- A passing score does not prove safety outside the tested scope.

Record these limitations with the release decision. Evaluation provides evidence for a bounded claim; it does not certify that an LLM system is correct in every situation.

## Practical takeaway

Evaluate the system at the level where decisions and failures occur. Start from the workload and its risks, construct representative slices, separate component failures, calibrate subjective graders, quantify uncertainty, and enforce quality, safety, latency, and cost gates together.

The most useful evaluation result is not “model A scored 0.82”. It is: “this version improves the intended workflow, stays within its operating budget, and does not regress on the failures we cannot accept.”

## References

1. Sainz, O. et al. (2023). [*NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark*](https://arxiv.org/abs/2310.18018). arXiv:2310.18018.
2. Deng, C. et al. (2023). [*Investigating Data Contamination in Modern Benchmarks for Large Language Models*](https://arxiv.org/abs/2311.09783). arXiv:2311.09783.
3. Es, S. et al. (2023). [*RAGAS: Automated Evaluation of Retrieval Augmented Generation*](https://arxiv.org/abs/2309.15217). arXiv:2309.15217.
4. Liu, X. et al. (2023). [*AgentBench: Evaluating LLMs as Agents*](https://arxiv.org/abs/2308.03688). arXiv:2308.03688.
5. Zheng, L. et al. (2023). [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685). arXiv:2306.05685.
6. Liang, P. et al. (2022). [*Holistic Evaluation of Language Models*](https://arxiv.org/abs/2211.09110). arXiv:2211.09110.
