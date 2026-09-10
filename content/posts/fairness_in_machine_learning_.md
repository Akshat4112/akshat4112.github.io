---
title: "Fairness in Machine Learning"
date: 2023-10-15T09:00:00+01:00
draft: false
tags: ["fairness", "machine-learning", "ethics", "ai", "bias"]
weight: 101
description: "An exploration of fairness in ML systems — defining group, individual, and causal fairness, sources of algorithmic bias, and techniques to address them."
showtoc: true
---

As machine learning systems are increasingly used in critical areas like finance, employment, and criminal justice, it's essential to ensure these models are fair and do not discriminate against certain groups. In this post, I will explore the concept of fairness in machine learning. Related discussions on algorithmic accountability often draw on tools such as [Hugging Face](https://huggingface.co/) model cards, which document bias evaluations for publicly released models.

## Defining Fairness

Fairness in machine learning can be understood in several ways:

- **Group Fairness**: This implies equal treatment or outcomes for different groups categorized by sensitive attributes like race or gender. For instance, ensuring a loan application system doesn't have a higher false rejection rate for one gender compared to another.

- **Individual Fairness**: This means that similar individuals should receive similar predictions or decisions, irrespective of their group membership. Two individuals with comparable financial backgrounds should get similar credit scores, regardless of their ethnicity or gender.

- **Causal Fairness**: Defined using causal modeling, it ensures similar predictions for individuals who would exhibit similar outcomes under different treatments. For example, a person's chances of getting a job should not be influenced by their gender.

## Sources of Unfairness

Unfairness in machine learning models can arise from several factors:

- **Biased Training Data**: If the training data reflects historical human biases, the model will likely inherit these biases.
- **Using Protected Variables**: Direct use of attributes like race or gender in models can lead to disparate treatment.
- **Proxy Variables**: Models may learn to discriminate using variables correlated with protected attributes, like zip codes.
- **Skewed Test Performance**: Poor model performance on minority groups due to imbalanced datasets.
- **Incorrect Similarity Metrics**: Discriminatory definitions of similarity between individuals can introduce bias.

## Techniques to Improve Fairness

Addressing unfairness involves strategies across the ML pipeline:

- **Pre-processing**: Removing biases in training data and identifying proxy variables.
- **In-processing**: Modifying the model training process to incorporate fairness constraints.
- **Post-processing**: Applying techniques post-training to correct biases.
- **Improved Evaluation**: Using specific metrics to assess fairness in different contexts.
- **Causal Modeling**: Employing [causal inference](https://en.wikipedia.org/wiki/Causal_inference) techniques to understand and mitigate biases.

## Worked example: inspect errors by group

Suppose a screening model is evaluated on two equally sized groups. The figures below are illustrative, not measurements from a deployed system.

| Group | True positives | False negatives | False positives | True negatives | TPR | FPR | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 36 | 4 | 12 | 48 | 90% | 20% | 48% |
| B | 24 | 16 | 6 | 54 | 60% | 10% | 30% |

The aggregate accuracy is not enough to reveal the difference. Group B has the lower false-positive rate, but also a much lower true-positive rate and selection rate. A team must decide which error matters in the application, examine uncertainty and sample size, and document the trade-off. Changing a threshold can improve one criterion while worsening another; there is no context-free fairness metric that resolves the policy decision automatically.

## Example: bias in digital recruitment advertising

A notable instance highlighting the need for fairness in AI was observed in digital recruitment advertising. An algorithm disproportionately showed high-salary job ads to men over women, influenced by biased historical data that reflected existing employment trends. This case underscores the importance of evaluating training data for biases and the necessity for ongoing algorithmic assessment to avoid reinforcing social inequalities.

## Conclusion

Achieving fairness in machine learning is a complex yet vital endeavor, requiring collaboration across various fields. With careful consideration and appropriate techniques, we can develop AI systems that are both ethical and equitable.
