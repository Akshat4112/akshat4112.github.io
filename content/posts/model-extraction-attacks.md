---
title: "Model Extraction Attacks: How Hackers Steal AI Models"
date: 2024-09-15T09:00:00+01:00
draft: false
tags: ["AI-security", "model-extraction", "machine-learning", "cybersecurity", "LLMs", "deep-learning", "AI"]
weight: 113
math: true
---

In the world of machine learning, especially with the rise of large language models (LLMs) and deep neural networks, model extraction attacks are a growing concern. These attacks aim to replicate the behavior of a machine learning model by querying it and then using the responses to reverse-engineer its underlying architecture and parameters.

## What is a Model Extraction Attack?

A **model extraction attack** occurs when an adversary tries to replicate a machine learning model by making repeated queries to it and analyzing its responses. The goal of the attacker is to create a new model that mimics the target model’s functionality, often without direct access to its architecture or parameters.

Once the adversary has successfully replicated a model, they can use it for various malicious purposes, including:

- **Stealing intellectual property**: By extracting a proprietary model, attackers can use it to bypass legal or technical protections or even re-sell it.
- **Bypassing security measures**: An extracted model might reveal vulnerabilities or ways to exploit the system, allowing attackers to bypass security checks.
- **Creating competitive advantages**: Competitors can replicate expensive and sophisticated models without having to invest in training them from scratch.

## Types of Model Extraction Attacks

There are two primary categories of model extraction attacks:

### 1. **Black-box Attacks**
In a **black-box** attack, the adversary only has access to the model’s input-output behavior, without any information about its internals. The attacker can make queries to the model and receive outputs, which they will use to infer the model’s behavior and architecture.

#### Example: 
Suppose an attacker wants to replicate a language model like GPT. They can send in a variety of text prompts to the model, such as "What’s the weather like today?" or "Tell me a joke," and observe the responses. After making enough queries, the attacker can train their own model on these input-output pairs, essentially trying to reproduce the target model's performance.

**Techniques used in Black-box Attacks**:
- **Querying the model extensively**: This is usually the most straightforward approach. The attacker queries the model with diverse inputs to gather enough data to approximate the model’s behavior.
- **Model Distillation**: The adversary can train a smaller surrogate model using the same input-output pairs. Although the extracted model will not match the target exactly, it can still replicate much of the functionality.

#### Real-World Example:
Researchers from the **University of California, Berkeley** demonstrated a black-box attack on image classification models. They used a model distillation technique where they queried a black-box image classifier (like Google Vision API) with thousands of images, extracting a surrogate model that performed similarly to the original model on classification tasks.

### 2. **White-box Attacks**
In a **white-box** attack, the adversary has full access to the model’s architecture, weights, and sometimes even its training data. This gives them an advantage in replicating the model, as they can directly inspect its components and behaviors.

#### Example:
If an attacker gains access to a model's source code or API endpoint (e.g., through a vulnerable cloud service), they can directly extract information about the model’s structure. This could include its layers, weights, and biases, making it much easier to create a replica.

**Techniques used in White-box Attacks**:
- **Exploiting exposed models**: If a company or service exposes their model without adequate protection (like an open-source model or poorly secured API), an attacker can directly replicate it.
- **Model stealing via backdoors**: Some attackers try to inject vulnerabilities into the model itself that would allow them to extract its parameters without permission.

#### Real-World Example:
In **NIPS 2016**, researchers successfully conducted a white-box model extraction attack on a neural network model by reverse-engineering the model architecture and retraining a copy of the model on their own data. This demonstrated the feasibility of stealing a model from an exposed API.

## Steps Involved in Model Extraction Attacks

### Step 1: Querying the Target Model
The adversary typically starts by querying the target model. In the case of black-box attacks, they don’t know the internal structure of the model, so they send a variety of queries, often including edge cases and adversarial inputs, to collect a wide range of responses.

**Example Implementation**:
```python
import requests

def query_model(input_text):
    response = requests.post('https://example.com/predict', data={'input': input_text})
    return response.json()

queries = [
    "What is 2+2?",
    "Tell me a story about a dragon.",
    "What is the capital of France?"
]

responses = [query_model(query) for query in queries]
```

In this case, the attacker collects responses from the target model and stores them for further analysis.

### Step 2: Analyzing the Responses
After gathering enough input-output pairs, the attacker will analyze the responses. They may look for patterns or anomalies that help them understand the model’s decision-making process.

For example, the attacker might notice that the model tends to classify certain types of input in a specific way, suggesting a particular feature in the underlying architecture.

### Step 3: Rebuilding the Model
In the final step, the attacker will attempt to train a new model using the gathered data. This process involves feeding the input-output pairs into a new model and adjusting its parameters until the model closely replicates the behavior of the target model.

**Example Implementation**:
```python
from sklearn.neural_network import MLPClassifier

# Example: Train a simple MLP classifier on the extracted data
X_train = [[1, 2], [2, 3], [3, 4]]  # Example inputs
y_train = [0, 1, 1]  # Example outputs (targets)

model = MLPClassifier(hidden_layer_sizes=(10,))
model.fit(X_train, y_train)

# Predict on new data
model.predict([[4, 5]])
```

This new model trained on the attacker’s collected data will likely approximate the behavior of the original model.

## Risks and Implications

The risks posed by model extraction attacks are vast:

- **Intellectual Property Theft**: Large companies invest a lot in developing machine learning models, and model extraction attacks make it easier for malicious actors to replicate their models, potentially leading to loss of competitive advantage.
- **Security Vulnerabilities**: Once an attacker has replicated a model, they could use it to exploit weaknesses or gain unauthorized access to sensitive data, especially if the model is used in mission-critical systems like finance or healthcare.
- **Reduction in Trust**: If model extraction attacks become more prevalent, users may lose trust in machine learning systems, fearing that adversaries could easily replicate and misuse models.

## Mitigating Model Extraction Attacks

### 1. **Limit Query Access**
Limiting the number of queries an external party can make to a model is a simple yet effective measure. Implementing rate limiting, CAPTCHA, or query restrictions can prevent an adversary from gathering enough data to replicate the model.

**Implementation Example**:
```python
import time
from functools import wraps

def limit_queries(rate_limit):
    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        def wrapped(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < rate_limit:
                time.sleep(rate_limit - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapped
    return decorator

@limit_queries(1)  # Only allow one query per second
def query_model(input_text):
    response = requests.post('https://example.com/predict', data={'input': input_text})
    return response.json()
```

### 2. **Model Watermarking**
Watermarking involves embedding unique markers within a model’s behavior. If an attacker replicates the model, these markers can be used to track and prove ownership.

### 3. **Obfuscating Model Outputs**
To make it harder for attackers to learn from the model’s behavior, you can introduce noise into the model’s outputs, making them less predictable and more difficult to replicate.

### 4. **Differential Privacy**
Differential privacy techniques can be applied to the model to ensure that individual data points cannot be reverse-engineered from the model’s responses. This reduces the effectiveness of model extraction attacks, as the model will not reveal sensitive information about specific data points.

## Final Thoughts

Model extraction attacks are a growing threat in the world of AI and machine learning. As models become more powerful and are deployed at scale, it’s crucial to understand the risks and employ countermeasures to protect intellectual property and secure sensitive systems.

By implementing appropriate defenses, including query limitations, model watermarking, and differential privacy, we can reduce the likelihood of successful model extraction and ensure that the benefits of AI are not overshadowed by malicious exploitation.

Stay tuned for more posts where we dive deeper into defense techniques and specific case studies.

— Akshat
