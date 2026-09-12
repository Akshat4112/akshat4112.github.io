---
title: "Model Extraction Attacks: Threat Models, Measurement, and Defences"
date: 2024-09-15T09:00:00+01:00
draft: false
tags: ["ai-security", "model-extraction", "machine-learning", "cybersecurity", "llm"]
weight: 113
math: true
showtoc: true
description: "A threat-model-first guide to model extraction: behavioural cloning, parameter recovery, query strategies, evaluation, detection, watermarking, privacy boundaries, and layered defences."
---

A prediction API protects model files from direct download, but it still exposes information about the model through its outputs. With enough carefully chosen queries, an adversary may train a substitute that imitates the service, infer properties of the target, or—in narrower settings—recover parameters.

These outcomes are often grouped under “model stealing”, although they are not equivalent. A surrogate that matches task accuracy may disagree with the target on many inputs. A high-fidelity copy need not contain the same parameters. Extracting memorised training records is a privacy attack, not proof that the model itself was copied.

A useful security assessment therefore begins with a precise asset, interface, attacker, and success criterion. Without those, query counts and surrogate scores are difficult to interpret.

## Define the threat before the attack

Let the deployed target be an oracle

$$
O(x; c) \rightarrow y,
$$

where \(x\) is the submitted input, \(c\) captures exposed controls such as temperature or requested output fields, and \(y\) may be a label, score vector, embedding, log probability, or generated sequence.

An attacker chooses queries using a strategy \(Q\). Query \(x_i\) may depend on previous observations:

$$
x_i \sim Q(\cdot \mid x_{<i}, y_{<i}, K_A),
$$

where \(K_A\) is the attacker’s prior knowledge: public data, the task definition, possible architecture families, or known training procedures. After a budget of \(B\) queries, the attacker produces an artefact \(g\).

The assessment must state at least:

- **Asset:** task behaviour, decision boundary, parameters, architecture, system prompt, or training records.
- **Access:** labels, confidence scores, logits, embeddings, generated text, timing, batch access, or gradients.
- **Control:** arbitrary inputs, partial feature vectors, decoding parameters, or only natural user requests.
- **Knowledge:** target family, feature representation, public data, or a related pretrained model.
- **Budget:** requests, tokens, accounts, money, wall-clock time, and local compute.
- **Goal:** fidelity to the target, independent task accuracy, parameter error, transferability, or recovered records.

“Black box” is not sufficiently precise. An API returning a class label presents a different attack surface from one returning full probabilities; a language-model endpoint that exposes token log probabilities differs from one returning only sampled text.

## Separate the neighbouring threats

The following attacks can share techniques, but they protect different assets.

| Threat | Attacker’s objective | Appropriate evidence |
|---|---|---|
| Functional extraction | Reproduce target behaviour on a distribution | Target–surrogate agreement on held-out inputs |
| Task-model replication | Build a model with comparable task utility | Accuracy or utility against ground truth |
| Parameter recovery | Recover specific weights or an equivalent parameterisation | Parameter or representation reconstruction error |
| Membership inference | Decide whether a record was in the training set | TPR at a stated FPR, with attack priors |
| Training-data extraction | Recover memorised training content | Verified unique records and precision |
| Prompt or configuration extraction | Recover hidden instructions or system configuration | Exact or semantic recovery under a defined test |
| System exfiltration | Obtain model files, credentials, or private data from infrastructure | Security incident evidence |

The last row is conventional application or infrastructure compromise. It should not be presented as query-based learning. Likewise, membership inference asks a one-bit question about a record’s presence; it does not attempt to clone the model. The original membership-inference formulation evaluates whether predictions reveal differences between members and non-members, while later work treats this as a hypothesis-testing problem.

For generative models, training-data extraction is also distinct. Carlini et al. recovered memorised sequences from GPT-2 through generation and ranking, demonstrating privacy leakage without constructing a replacement model.

## What extraction success means

Suppose the attacker trains a surrogate \(g\). Two metrics answer different questions.

**Fidelity** measures agreement with the target:

$$
\operatorname{Fid}_{P}(g,O)
= \mathbb{E}_{x\sim P}
\left[\mathbf{1}\{g(x)=O(x)\}\right].
$$

**Task accuracy** measures correctness against ground-truth labels \(y\):

$$
\operatorname{Acc}_{P}(g)
= \mathbb{E}_{(x,y)\sim P}
\left[\mathbf{1}\{g(x)=y\}\right].
$$

A surrogate can have good task accuracy but low fidelity if it makes different mistakes from the target. It can also achieve high fidelity on common inputs while failing near decision boundaries or on important subgroups.

For probabilistic classifiers, report a distributional distance such as KL divergence or mean absolute difference in scores, provided the API exposes probabilities. For language models, exact string match is usually too brittle. Evaluation may combine task-specific quality, behavioural agreement under controlled decoding, output-distribution measures where log probabilities are available, and targeted probes of safety or style. The metric must follow the stated asset.

Always evaluate on held-out inputs that were not used to build the transfer set. Include both the expected production distribution and security-relevant slices. Report confidence intervals and compare with a baseline trained on the attacker’s public data without oracle labels.

## How query-based extraction works

A common functional-extraction pipeline creates a transfer set:

$$
D_Q = \{(x_i, O(x_i))\}_{i=1}^{B},
$$

then trains a surrogate

$$
\hat{g}
= \arg\min_{g\in \mathcal{G}}
\frac{1}{B}\sum_{i=1}^{B}
\ell(g(x_i), O(x_i)).
$$

This resembles knowledge distillation, but the access, consent, and objective differ. If the target returns a score vector \(p_O(x)\), a student may minimise

$$
\mathcal{L}_{\text{distil}}
= T^2 D_{\mathrm{KL}}
\left(p_O^{(T)}(x)\,\|\,p_g^{(T)}(x)\right),
$$

where \(T\) is a chosen temperature and both distributions must be defined consistently. An attacker cannot retroactively change the target’s output temperature unless the interface permits it.

Query selection can be:

- **natural:** sample from public data close to expected use;
- **adaptive:** choose inputs from earlier target–surrogate disagreement or surrogate uncertainty;
- **synthetic:** optimise or generate inputs to expose informative behaviour;
- **task-specific:** exploit knowledge of the model family or feature representation.

Uncertainty sampling,

$$
x_i = \arg\max_{x\in U} H(g(x)),
$$

is one possible heuristic over a candidate pool \(U\). It is not universally optimal: surrogate uncertainty may be poorly calibrated, the pool may miss the target distribution, and anomaly detection may flag synthetic query patterns.

## Query complexity has no universal formula

Claims such as “\(O(d)\) queries extract a linear model” or “\(O(p)\) queries extract a neural network” omit essential assumptions. Query requirements depend on:

- whether outputs are exact equations, probabilities, rounded scores, or labels;
- whether inputs can be selected arbitrarily or only sampled;
- whether the feature encoding and model family are known;
- whether the goal is exact parameter recovery, functional fidelity, or task accuracy;
- the evaluation distribution and desired error;
- noise, randomised decoding, and adaptive defences.

Tramèr et al. demonstrated equation-solving attacks with near-perfect fidelity for specific model classes and API behaviours. Those results are concrete attack constructions, not general bounds for every linear classifier or neural network.

Standard learning-theory bounds are also easy to misuse here. A statement involving VC dimension needs an i.i.d. sampling assumption, a defined hypothesis class and loss, and a relationship between the sampling distribution and the evaluation distribution. Adaptive queries and target-provided labels do not automatically satisfy those conditions.

The defensible empirical presentation is a query–fidelity curve:

$$
F(B) = \operatorname{Fid}_{P}(g_B,O),
$$

reported across budgets, repeated attack runs, and attacker strategies. Include monetary and computational costs, because two attacks using the same number of API calls may consume very different numbers of tokens or local training resources.

## LLM-specific extraction

Generative APIs widen the definition of “output”. Each response can reveal generated tokens, refusal patterns, formatting habits, log probabilities, embeddings, or latent configuration.

Functional imitation of an LLM generally means training another model on prompts and target-generated responses. The resulting model may mimic selected behaviours without copying the target’s weights. Claims of “stealing the LLM” should identify which behaviours were reproduced, on which distribution, and at what cost.

Parameter recovery is narrower but possible under unusually informative interfaces. Carlini et al. showed that exposed log probabilities could be used to recover an embedding projection layer of particular production language models, up to symmetries. This does not imply that ordinary text-only access permits recovery of every transformer parameter.

Randomised decoding does not by itself prevent extraction. It changes the observations and may require repeated queries or distributional evaluation. Conversely, variability can make superficial string agreement a poor fidelity metric.

## Defences must match the threat model

No individual control prevents every extraction objective. A useful defence plan combines exposure reduction, economic friction, detection, ownership evidence, and ordinary infrastructure security.

### Minimise output exposure

Return only what legitimate clients need. Removing confidence vectors, excessive precision, token log probabilities, embeddings, diagnostic traces, or unrestricted sampling controls can reduce information per query.

This is risk reduction, not a guarantee. Tramèr et al. showed that attacks can remain possible when confidence values are omitted. Output reduction may also damage legitimate calibration, ranking, explainability, or research use, so measure utility before deployment.

### Authenticate, meter, and rate-limit

Associate calls with identities, quotas, cost, and purpose. Rate limits raise the time and account cost of high-volume attacks, but a patient or distributed attacker may evade per-account thresholds. Enforce organisation- and payment-level limits where appropriate, and protect account creation from abuse.

### Detect suspicious query sequences

Detection should consider sequence-level signals such as unusual input distances, repeated boundary probing, high coverage, synthetic distributions, and coordinated identities. PRADA, for example, detects shifts in distances between consecutive queries for the attacks and datasets evaluated in that work.

A reported zero false-positive count on a study dataset is not a universal zero false-positive guarantee. Validate detectors against the service’s real benign traffic, adaptive attackers, model updates, and seasonal changes. Use alerts to trigger investigation or graduated throttling rather than treating a research detector as proof of malicious intent.

### Perturb or round outputs carefully

Rounding scores, reducing precision, randomising responses, or adding noise may reduce usable signal, but the effect depends on the attack and can often be averaged out through repeated queries. The defence may also harm calibration and reproducibility.

Do not claim an extraction-error floor of \(\Omega(\sigma^2)\) from Gaussian output noise without a stated model, loss, estimator, independence assumption, and query budget. Treat perturbation as an empirical security–utility trade-off and test adaptive repeat-query attacks.

### Use watermarking or fingerprinting as evidence

Watermarks and fingerprints usually support post hoc ownership verification; they do not necessarily stop extraction. A trigger-set watermark can test whether a suspected surrogate reproduces unusual target behaviour.

False-positive probability is not simply \(|\mathcal{Y}|^{-k}\) unless trigger responses are independent, uniformly distributed, secret, and tested with a fixed rule. Real predictions and triggers can be correlated, and testing choices can inflate error rates. Pre-register the verification procedure, evaluate unrelated models, report false-positive and false-negative rates, and account for multiple comparisons.

Watermarks can also be removed, evaded, forged, or themselves inferred. For LLM text watermarks, watermark-stealing research has demonstrated spoofing and scrubbing attacks. Describe watermark evidence as statistical and scheme-specific, not “cryptographic-strength provenance”.

### Protect infrastructure separately

Encrypt model artefacts, control deployment permissions, isolate serving systems, rotate credentials, monitor downloads, and restrict debug endpoints. These controls address direct exfiltration and insider threats. They are essential, but they do not stop an authorised caller from learning through permitted outputs.

## Where differential privacy fits

Differential privacy protects the contribution of individual training records. For a randomised training algorithm \(M\), \((\varepsilon,\delta)\)-DP requires that for neighbouring datasets \(D\) and \(D'\),

$$
\Pr[M(D)\in S]
\leq e^{\varepsilon}\Pr[M(D')\in S] + \delta
$$

for every measurable set \(S\).

This guarantee is relevant to membership inference and training-data privacy. It does not generally prevent a caller from training a surrogate that matches the released model’s public behaviour. A differentially private model may still be easy to imitate, because functional extraction targets the model mapping rather than one person’s presence in the training set.

DP therefore belongs in the training-data privacy plan, with an explicit privacy accountant, adjacency definition, and utility evaluation. It should not be listed as a universal defence for model intellectual property.

## A reproducible extraction assessment

Use a pre-registered protocol rather than selecting the most favourable attack result afterwards.

1. **Define the target asset and interface.** Record every observable output and controllable parameter.
2. **Specify attacker knowledge and budget.** Include public datasets, pretrained checkpoints, identities, tokens, cost, and compute.
3. **Choose baselines.** Train the same surrogate without oracle labels and include a legitimate-use workload.
4. **Separate query and evaluation data.** Keep the held-out fidelity set inaccessible to query selection.
5. **Run multiple strategies and seeds.** Report variation rather than a single best run.
6. **Measure by budget.** Plot fidelity, task utility, and attack cost against requests and tokens.
7. **Test meaningful slices.** Include common inputs, rare classes, boundary cases, subgroups, and adversarial examples where relevant.
8. **Enable defences individually and jointly.** Measure both attack reduction and legitimate-user degradation.
9. **Record detector operating points.** Report TPR at stated FPRs, not only accuracy.
10. **Document limitations.** State what the experiment cannot establish about parameters, ownership, or training data.

A compact result table might look like this:

| Configuration | Queries | Cost | Fidelity | Task accuracy | Detector TPR at 1% FPR | Legitimate utility |
|---|---:|---:|---:|---:|---:|---:|
| Public-data baseline | 0 | Local only | — | Report | — | — |
| Natural queries | Report | Report | Report | Report | Report | Report |
| Adaptive queries | Report | Report | Report | Report | Report | Report |
| Adaptive + defence | Report | Report | Report | Report | Report | Report |

For LLMs, replace task accuracy with task-specific quality metrics and include token volume, decoding settings, prompt families, safety behaviour, and repeated samples.

## A precise security objective

Extraction is not accurately represented by

$$
\min_g \max_Q \mathbb{E}[\ell(g,O)],
$$

because an attacker tries to minimise disagreement while choosing a query strategy that improves learning, and a defender controls a different set of mechanisms. A clearer empirical objective for the attacker is

$$
\min_{Q,A}
R_P\!\left(A(T_B(Q,O)), O\right)
\quad
\text{subject to }
C(Q,A) \leq B,
$$

where \(T_B\) is the transcript collected under budget \(B\), \(A\) trains the surrogate, \(R_P\) is disagreement on evaluation distribution \(P\), and \(C\) accounts for queries, tokens, money, and compute.

The defender does not simply maximise this loss. The defender selects controls \(d\) that reduce extraction while preserving legitimate utility:

$$
\min_d
\left(
\operatorname{Risk}_{\text{extract}}(d)
+ \lambda\,\operatorname{UtilityLoss}(d)
+ \mu\,\operatorname{OperationalCost}(d)
\right).
$$

Neither expression supplies a universal certified bound. They make the objectives and trade-offs explicit enough to design an experiment.

## Takeaway

Model extraction is not one attack with one query bound or one defence. It is a family of threats whose feasibility depends on what the API reveals, what the attacker already knows, what “copying” means, and how success is measured.

The strongest production response is layered: minimise unnecessary outputs, authenticate and meter access, detect suspicious sequences, preserve ownership evidence, secure infrastructure, and evaluate each control against adaptive attacks and legitimate-user utility. Keep privacy guarantees separate from intellectual-property claims, and avoid calling behavioural similarity proof of parameter theft.

## References

1. Tramèr, F. et al. (2016). [Stealing Machine Learning Models via Prediction APIs](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer). 25th USENIX Security Symposium.
2. Orekondy, T., Schiele, B., and Fritz, M. (2019). [Knockoff Nets: Stealing Functionality of Black-Box Models](https://arxiv.org/abs/1812.02766). CVPR 2019.
3. Juuti, M. et al. (2019). [PRADA: Protecting Against DNN Model Stealing Attacks](https://arxiv.org/abs/1805.02628). IEEE European Symposium on Security and Privacy.
4. Shokri, R. et al. (2017). [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820). IEEE Symposium on Security and Privacy.
5. Carlini, N. et al. (2021). [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805). 30th USENIX Security Symposium.
6. Carlini, N. et al. (2024). [Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634). arXiv:2403.06634.
7. Jovanović, N., Staab, R., and Vechev, M. (2024). [Watermark Stealing in Large Language Models](https://arxiv.org/abs/2402.19361). ICML 2024.
