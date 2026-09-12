---
title: "Fairness in Machine Learning: Metrics, Trade-offs, and Evaluation"
date: 2023-10-15T09:00:00+01:00
lastmod: 2026-09-12T23:58:00+02:00
draft: false
tags: ["fairness", "machine-learning", "ethics", "ai", "bias"]
weight: 101
description: "A practical guide to defining, measuring, and monitoring fairness in machine-learning systems without hiding the policy choices behind one metric."
showtoc: true
math: true
---

Fairness is not a property that can be established by checking whether one metric exceeds a universal threshold. It is a system-level question about people, decisions, benefits, harms, institutions, and the evidence used to justify an intervention.

Machine-learning metrics still matter. They can reveal differences in selection rates, error rates, calibration, and performance across groups. But choosing a metric is already a policy decision: it determines which differences count as harms and which trade-offs receive attention.

This article develops a practical evaluation framework for classification and ranking systems. It does not provide legal advice, and statistical parity should not be treated as proof of legal or ethical compliance.

## Start with the decision, not the model

Before calculating a fairness metric, define the system around the model:

1. **Decision:** What action is taken from the prediction?
2. **Population:** Who can enter the system, and who is excluded before scoring?
3. **Outcome:** What real-world construct is the label intended to represent?
4. **Harm:** Who bears false positives, false negatives, delays, denials, or surveillance?
5. **Benefit:** Who receives access, review, assistance, or opportunity?
6. **Recourse:** Can a person contest or correct the data and decision?
7. **Alternatives:** Could the objective be met without prediction or with a less intrusive process?

A model can have similar error rates across groups while the surrounding process remains unfair. For example, document-quality failures may prevent some applicants from reaching the model, a review queue may systematically delay one language group, or the target label may encode an unequal historical process.

The unit of analysis must therefore include data collection, model development, threshold selection, human review, appeals, and downstream outcomes.

## Notation for binary decisions

Let:

- $Y \in \{0,1\}$ be the observed outcome;
- $\hat{Y} \in \{0,1\}$ be the model-assisted decision;
- $S \in [0,1]$ be a score; and
- $A$ be a protected or policy-relevant group attribute.

For a group $a$:

\[
\operatorname{TPR}_a=P(\hat{Y}=1\mid Y=1,A=a),
\]

\[
\operatorname{FPR}_a=P(\hat{Y}=1\mid Y=0,A=a),
\]

\[
\operatorname{PPV}_a=P(Y=1\mid \hat{Y}=1,A=a).
\]

These quantities describe different questions. The true-positive rate measures how often positive cases are selected; the false-positive rate measures how often negative cases are incorrectly selected; positive predictive value measures how often selected cases have a positive observed outcome.

## Major fairness criteria

### Demographic parity

Demographic parity requires equal selection rates:

\[
P(\hat{Y}=1\mid A=a)=P(\hat{Y}=1\mid A=b).
\]

It focuses on allocation across groups without conditioning on the observed label. This may be relevant when selection itself is the benefit or burden. It can be inappropriate when legitimate outcome differences matter, and it can conceal unequal error types.

### Equal opportunity

Equal opportunity requires equal true-positive rates:

\[
P(\hat{Y}=1\mid Y=1,A=a)=P(\hat{Y}=1\mid Y=1,A=b).
\]

It asks whether people with a positive observed outcome have the same chance of receiving a positive decision. The criterion prioritises false negatives and may be suitable when a missed positive creates the central harm.

### Equalised odds

Equalised odds requires both equal true-positive and false-positive rates:

\[
\hat{Y}\ \perp\ A\mid Y.
\]

This constrains error rates conditional on the observed outcome. It is more demanding than equal opportunity, but still depends on whether $Y$ is a defensible label.

### Predictive parity

Predictive parity requires comparable positive predictive value:

\[
P(Y=1\mid \hat{Y}=1,A=a)=P(Y=1\mid \hat{Y}=1,A=b).
\]

It asks whether a positive decision has a similar interpretation across groups. It does not ensure equal access to positive decisions or equal error rates.

### Calibration within groups

A score is calibrated within groups when, for score value $s$,

\[
P(Y=1\mid S=s,A=a)=s.
\]

Among people assigned a score of 0.7, roughly 70% should have the observed positive outcome in each evaluated group. Calibration concerns the meaning of a risk score; it does not determine whether the same threshold should be used or whether the underlying label is fair.

### Individual fairness

Individual fairness proposes that similar individuals should receive similar outputs. Dwork and colleagues formalised this idea using a task-specific similarity metric:

\[
D(M(x),M(x'))\leq d(x,x').
\]

The difficult part is defining $d$: deciding who is similar is a substantive domain judgement and can reproduce existing inequalities.

### Counterfactual fairness

Counterfactual fairness uses a causal model to ask whether a decision would remain unchanged in a counterfactual world where the protected attribute differed. It requires explicit causal assumptions, not merely removing the protected column. Those assumptions are often contested and cannot be validated from observational data alone.

## Why fairness criteria can conflict

Several desirable criteria cannot generally be satisfied at the same time when outcome prevalence differs across groups and prediction is imperfect. Kleinberg, Mullainathan, and Raghavan and, independently, Chouldechova formalised incompatibilities between calibration or predictive parity and error-rate parity.

The intuition comes from Bayes' rule. For group $a$ with prevalence $\pi_a=P(Y=1\mid A=a)$:

\[
\operatorname{PPV}_a=
\frac{\operatorname{TPR}_a\pi_a}
{\operatorname{TPR}_a\pi_a+\operatorname{FPR}_a(1-\pi_a)}.
\]

If two groups have different prevalences but identical TPR and FPR, their PPVs will normally differ. Equalising one family of metrics can therefore move another away from parity.

This is not a reason to abandon measurement. It means a team must document which harms and rights determine the chosen criterion, rather than presenting the selected metric as mathematically inevitable.

## Worked example: the same accuracy, different harms

Consider a hypothetical triage model evaluated on two groups of 100 cases each. The values are illustrative.

| Group | TP | FN | FP | TN | Accuracy | TPR | FPR | Selection rate | PPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 36 | 4 | 12 | 48 | 84% | 90% | 20% | 48% | 75% |
| B | 24 | 16 | 6 | 54 | 78% | 60% | 10% | 30% | 80% |

An aggregate result can obscure several differences:

- Group B has a 30 percentage-point lower TPR, so more positive cases are missed.
- Group A has a 10 percentage-point higher FPR, so more negative cases are incorrectly selected.
- Group A has a higher selection rate.
- Group B has a slightly higher PPV despite its lower TPR.

There is no context-free answer to which row is “fairer”. If selection triggers beneficial specialist review, missed positives may dominate. If selection creates an intrusive investigation, false positives may dominate. If the observed label reflects unequal access to earlier services, conditioning on it may reinforce the problem.

The review should add uncertainty intervals. With small subgroups, a few cases can move rates substantially. Report numerator and denominator alongside percentages, and avoid ranking groups from noisy point estimates.

## Thresholds are policy choices

A scoring model becomes a decision system when a threshold or ranking rule is applied. Moving the threshold changes TPR, FPR, precision, workload, and the distribution of benefits or burdens.

Group-specific thresholds can sometimes satisfy a statistical criterion, but they introduce legal, ethical, operational, and communication questions. A technical team should not select them in isolation. The decision should involve domain experts, affected stakeholders, risk or compliance functions, and legal review where appropriate.

Evaluate threshold candidates with:

- confusion matrices and uncertainty by group;
- expected harm or utility for each error type;
- review capacity and waiting time;
- sensitivity to prevalence shifts;
- calibration and decision curves;
- recourse and appeal outcomes; and
- performance at intersections, not only broad categories.

## Where unfairness enters the pipeline

### Historical and sampling bias

Training data reflects who was observed and how institutions acted. Under-representation can increase variance for smaller groups, while historical decisions can become labels that reproduce an earlier policy.

### Measurement and label bias

A convenient label may be a poor proxy for the construct of interest. Healthcare expenditure, arrests, repayment history, and manager ratings can reflect unequal access or enforcement rather than underlying need, behaviour, or ability.

### Real-world case: healthcare allocation

Obermeyer and colleagues examined a widely used US healthcare-management algorithm that used predicted future healthcare cost as a proxy for health need. At the same risk score, Black patients were substantially sicker than White patients. Replacing predicted cost with a measure of health need would have increased the proportion of Black patients selected for additional care from 17.7% to 46.5% ([Obermeyer et al., 2019](https://doi.org/10.1126/science.aax2342)).

The case demonstrates how a seemingly neutral target can create disparity even when race is not an explicit model input: unequal access to care and unequal spending made cost an imperfect proxy for need. It does not establish that every healthcare algorithm or cost-based model has the same failure. The study concerned a particular commercial population-health algorithm and setting, so teams must test label validity, access patterns, and group outcomes in their own deployment context.

### Feature and proxy effects

Removing a protected attribute does not remove its information. Location, employment history, language, device, and other variables can act as proxies. Conversely, retaining protected attributes may be necessary to evaluate disparities. Data access should be purpose-limited and governed rather than decided by a simplistic “fairness through unawareness” rule.

### Model and optimisation choices

An aggregate loss gives more influence to common patterns. Regularisation, representation learning, thresholding, and hyperparameter selection can affect groups differently even when the code does not use explicit protected categories.

### Interface and automation bias

Human reviewers may over-trust a score, misunderstand uncertainty, or treat a recommendation as mandatory. A nominal “human in the loop” does not reduce harm unless reviewers have authority, context, time, and a meaningful way to disagree.

### Deployment and feedback loops

Model decisions influence future data. Additional investigation can create more recorded adverse events for one group, while denied applicants may disappear from outcome data. Monitoring only cases that received a positive decision creates selective-label problems.

## Intersectional and slice-based evaluation

Broad averages such as “women” or “customers over 60” can hide failures at intersections. Evaluate relevant combinations of attributes, geography, language, document type, channel, and operational conditions.

This creates a multiple-comparisons and sample-size problem. A practical approach is to:

1. predefine high-risk slices from domain knowledge;
2. discover additional slices as a diagnostic step;
3. report counts and uncertainty;
4. avoid publishing identifiable results for very small groups;
5. confirm findings on independent data; and
6. use hierarchical estimates or pooled evidence when justified.

Subgroup analysis should not become a fishing exercise in which only the most alarming point estimate is reported.

## Mitigation strategies

Mitigation can occur before, during, or after model training, but each intervention changes something concrete and should be evaluated for side effects.

| Stage | Examples | Main caution |
|---|---|---|
| Problem formulation | Change the target, decision, or use of prediction | A better model cannot repair an unjust objective |
| Data collection | Improve coverage, labels, and measurement | More data can reproduce the same measurement bias |
| Pre-processing | Reweight or resample observations | May affect calibration and variance |
| In-processing | Add constraints or group-robust objectives | Optimises the selected formal criterion, not fairness generally |
| Post-processing | Adjust thresholds or decisions | Requires governance and may create group-specific treatment |
| Product process | Add review, recourse, or alternative evidence | Human processes can introduce new disparities |

Do not begin with a mitigation technique. Begin with the harm model, then test whether the intervention reduces that harm without unacceptable losses elsewhere.

## A production evaluation plan

A credible fairness review is reproducible and tied to operational decisions.

### Before deployment

- Define intended use, excluded uses, population, decision owner, and affected parties.
- Document label construction and known measurement limitations.
- Establish a strong non-ML or simpler-model baseline.
- Report overall and subgroup performance with counts and uncertainty.
- Evaluate threshold alternatives and the distribution of error costs.
- Test missingness, out-of-distribution inputs, and upstream extraction failures by group.
- Conduct qualitative review with domain experts and people affected by the process.
- Document residual risks, recourse, escalation, and rollback criteria.

### After deployment

- Monitor input coverage, missingness, score distributions, decisions, errors, and delays by relevant slice.
- Separate model drift from policy, population, and data-pipeline changes.
- Audit overrides: who changes model recommendations, in which direction, and with what outcome?
- Track complaints, appeals, corrections, and reversals as first-class signals.
- Reassess metrics when the decision process or definition of harm changes.
- Use incident review and rollback procedures for material disparities.

Model cards and system documentation can make intended use, subgroup results, and limitations visible, but documentation is evidence of a process—not a substitute for governance.

## Fairness for generative AI and document systems

Generative systems require additional evaluation beyond a binary confusion matrix. Relevant slices may include language, dialect, document quality, handwriting, name origin, accessibility needs, and subject matter.

For an OCR-to-LLM workflow, test each stage:

| Stage | Example fairness question |
|---|---|
| Intake | Are some users more likely to submit unsupported formats? |
| OCR | Does character or field error rate differ by language, script, scan quality, or handwriting? |
| Retrieval | Is relevant evidence recalled equally across document types and groups? |
| Generation | Are unsupported claims, refusals, or harmful stereotypes unevenly distributed? |
| Human review | Do queue time, override rate, and escalation differ across slices? |
| Outcome | Do errors translate into unequal delays, denials, or burdens? |

Evaluate the final decision and intermediate components. A parity result at the language-model layer can be meaningless if the OCR system fails earlier for one group.

## Limitations

Protected attributes may be unavailable, legally restricted, inaccurately recorded, or socially constructed in ways that resist fixed categories. Small sample sizes limit inference. Observed labels may not represent the outcome society actually values. Metrics measured offline may not predict harms after deployment.

Formal criteria also omit procedural questions: Was the use of prediction legitimate? Were people informed? Can they challenge the decision? Who is accountable? Was a less harmful alternative available?

Fairness evaluation can identify disparities and clarify trade-offs. It cannot convert a contested social decision into a purely technical optimisation problem.

## Practical takeaway

Treat fairness as an evidence-backed governance process:

1. define the decision, population, harms, benefits, and recourse;
2. inspect data and labels before selecting a model;
3. choose metrics that correspond to the most important harms;
4. report multiple metrics, counts, and uncertainty by relevant intersectional slices;
5. make threshold and mitigation choices explicit;
6. test the complete human and technical workflow; and
7. monitor downstream outcomes, appeals, and feedback loops after deployment.

The goal is not to declare a model fair. It is to make disparities measurable, policy choices visible, and corrective action possible.

## References

1. Barocas, S., Hardt, M. and Narayanan, A. [Fairness and Machine Learning: Limitations and Opportunities](https://fairmlbook.org/). 2023.
2. Dwork, C. et al. [Fairness Through Awareness](https://doi.org/10.1145/2090236.2090255). *Innovations in Theoretical Computer Science*, 2012.
3. Hardt, M., Price, E. and Srebro, N. [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413). *NeurIPS*, 2016.
4. Kleinberg, J., Mullainathan, S. and Raghavan, M. [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807). *ITCS*, 2017.
5. Chouldechova, A. [Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments](https://doi.org/10.1089/big.2016.0047). *Big Data*, 2017.
6. Kusner, M. J. et al. [Counterfactual Fairness](https://arxiv.org/abs/1703.06856). *NeurIPS*, 2017.
7. Suresh, H. and Guttag, J. [A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle](https://doi.org/10.1145/3465416.3483305). *Equity and Access in Algorithms, Mechanisms, and Optimization*, 2021.
8. Selbst, A. D. et al. [Fairness and Abstraction in Sociotechnical Systems](https://doi.org/10.1145/3287560.3287598). *FAT\**, 2019.
9. Mitchell, M. et al. [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596). *FAT\**, 2019.
10. NIST. [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://doi.org/10.6028/NIST.AI.100-1). 2023.
11. Obermeyer, Z. et al. [Dissecting racial bias in an algorithm used to manage the health of populations](https://doi.org/10.1126/science.aax2342). *Science*, 2019.
