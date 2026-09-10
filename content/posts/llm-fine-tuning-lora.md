---
title: "Fine-Tuning LLMs with LoRA"
date: 2025-04-20T17:30:00+02:00
lastmod: 2026-09-10T00:00:00+00:00
draft: false
tags: ["llm", "fine-tuning", "lora", "qlora", "preference-learning"]
weight: 105
description: "A practical guide to deciding when an LLM needs fine-tuning, understanding LoRA and QLoRA, and building an evaluated PEFT training workflow."
math: true
showtoc: true
---

Fine-tuning is useful when a model repeatedly exhibits the wrong behaviour and high-quality examples can define the behaviour you want. It is not the default way to add changing facts, private documents, or authoritative records to an application.

That distinction matters. Training a model on policy documents may make its language sound familiar, but it does not provide reliable document versioning, access control, citations, or guaranteed recall. Retrieval-augmented generation (RAG) or a deterministic tool is usually a better interface to knowledge that must remain inspectable.

Low-Rank Adaptation (LoRA) makes weight adaptation substantially more practical by freezing the pretrained model and learning small low-rank updates [1]. This article explains the mathematics, the engineering trade-offs, and an evaluation-led workflow for deciding whether LoRA is the right intervention.

## Start with the failure, not the method

Before creating a training dataset, identify what fails under a strong baseline prompt and representative evaluation set.

| Observed need | Start with | Why |
|---|---|---|
| Current or private facts with citations | RAG | Knowledge remains external, updateable, and inspectable |
| Calculation or database lookup | Tool or API | The authoritative operation is deterministic |
| Stable output schema | Prompting plus constrained decoding | Often sufficient without changing weights |
| Repeated task or formatting errors | Supervised fine-tuning (SFT) | Demonstrations can teach the desired mapping |
| Tone or domain-language adaptation | SFT, often with LoRA | Behaviour is present in curated target responses |
| Preference between two valid responses | Preference optimisation such as DPO | Pairwise data expresses relative quality |
| Broad capability change with ample compute | Full fine-tuning | Maximum adaptation capacity, with higher cost and risk |

Use the least invasive method that fixes the measured failure. A prompt change is easier to inspect and reverse than a training run. RAG is easier to refresh than facts encoded in weights. Fine-tuning becomes justified when these simpler controls plateau and the remaining errors are learnable from examples.

## What supervised fine-tuning changes

For an input sequence \(x\) and desired response \(y=(y_1,\ldots,y_T)\), causal-language-model SFT minimises the negative log-likelihood of the target tokens:

\[
\mathcal{L}_{\text{SFT}}
=-\sum_{t=1}^{T}\log p_{\theta}(y_t \mid x,y_{<t}).
\]

The training signal says, in effect, “produce this response for this kind of input”. Dataset quality therefore dominates the result:

- instructions must match the production task;
- target responses must demonstrate the complete desired behaviour;
- examples must cover boundary and refusal cases;
- formatting must match the model's chat template; and
- train, validation, and release sets must not leak duplicates.

SFT does not automatically teach truthfulness. If targets contain unsupported claims or inconsistent style, the model learns those patterns as well.

## How LoRA represents the update

Consider a frozen linear layer with:

\[
W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
\]

and input:

\[
x \in \mathbb{R}^{d_{\text{in}}}.
\]

Full fine-tuning learns a dense update \(\Delta W\) with
\(d_{\text{out}}d_{\text{in}}\) parameters. LoRA constrains that update to a product of two smaller matrices:

\[
A \in \mathbb{R}^{r\times d_{\text{in}}}, \qquad
B \in \mathbb{R}^{d_{\text{out}}\times r},
\]

where \(r \ll \min(d_{\text{in}},d_{\text{out}})\). The adapted forward pass is:

\[
h = W_0x + \frac{\alpha}{r}BAx.
\]

Here:

- \(W_0\) remains frozen;
- \(A\) and \(B\) are trainable;
- \(r\) controls the rank and adapter capacity; and
- \(\alpha/r\) scales the learned update.

The adapter contributes

\[
r(d_{\text{in}}+d_{\text{out}})
\]

trainable parameters instead of \(d_{\text{in}}d_{\text{out}}\).

For a square \(4096 \times 4096\) projection, full adaptation would update 16,777,216 weights. A rank-16 LoRA update trains:

\[
16(4096+4096)=131{,}072
\]

parameters, about 0.78% of that projection's dense weight count. The model still has to execute the frozen layer; LoRA mainly reduces trainable parameters, optimiser state, gradient storage, and checkpoint size. Activation memory and base-model inference memory do not disappear.

The original LoRA paper applied low-rank updates to transformer weight matrices and reported competitive results on its evaluated models and tasks [1]. That evidence does not establish that one rank or target-module choice works universally.

## Rank, scaling, and target modules

Three choices determine much of a LoRA experiment.

### Rank

A larger rank increases adapter capacity and parameter count. Too little capacity can underfit; more rank can add cost without improving the target metric. Sweep rank against a fixed evaluation set rather than treating 8, 16, or 64 as a rule.

### Scaling and dropout

\(\alpha\) controls the scale of the adapter contribution relative to the frozen path. LoRA dropout regularises the adapter input during training. Both interact with learning rate, data size, and target modules, so record them as part of the model artefact.

### Target modules

Attention query and value projections were common targets in early LoRA experiments. Modern causal models also contain key, output, gate, up, and down projections that may benefit from adaptation. Module names vary by architecture.

Inspect model.named_modules() and confirm what the configuration matched. A run that silently targets the wrong or too few layers can look valid while training an unintended model.

## LoRA and QLoRA are not the same thing

LoRA describes the low-rank parameterisation. QLoRA is a training approach that keeps the pretrained model frozen in 4-bit quantised form and backpropagates through it into LoRA adapters [2].

The QLoRA work introduced:

- 4-bit NormalFloat (NF4) for normally distributed pretrained weights;
- double quantisation of quantisation constants; and
- paged optimisers to manage memory spikes.

QLoRA can make a larger base model fit into limited accelerator memory, but it does not mean every computation occurs in 4-bit precision. The adapter and compute dtypes are separate choices. Quantisation also introduces another source of approximation that must be evaluated on the target task.

QLoRA is therefore a method, not a library. Libraries such as Transformers, bitsandbytes, PEFT, and TRL can be used to implement it.

## SFT and DPO optimise different data

SFT consumes target responses. Direct Preference Optimisation (DPO) consumes pairs:

\[
(x, y_{\text{chosen}}, y_{\text{rejected}}).
\]

DPO directly optimises a policy against preferences relative to a reference policy, avoiding the separate learned reward model and reinforcement-learning loop used in conventional RLHF pipelines [3]. It is appropriate when reviewers can reliably say which of two responses is better, even when writing one canonical target is difficult.

A common workflow is:

1. start from a capable base or instruction-tuned model;
2. run SFT on high-quality demonstrations;
3. collect chosen/rejected pairs around remaining quality differences;
4. apply DPO or another justified preference objective; and
5. compare every stage against the unchanged baseline.

Do not apply DPO merely because preference data exists. Inspect how pairs were generated, whether the preference is unambiguous, whether response length leaks the label, and whether the data represents production decisions.

## A reproducible PEFT example

The example below performs LoRA SFT on a small causal model using current PEFT and TRL interfaces. Pin dependency and model revisions in a real experiment; library APIs and model files can change.

~~~bash
pip install "transformers>=4.57" "peft>=0.17" "trl>=0.26" \
  "datasets>=4.0" "accelerate>=1.10"
~~~

~~~python
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

examples = [
    {
        "messages": [
            {"role": "user", "content": "Rewrite: Payment shall be remitted within 30 days."},
            {"role": "assistant", "content": "Please pay within 30 days."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Rewrite: The applicant must furnish documentation."},
            {"role": "assistant", "content": "The applicant must provide the documents."},
        ]
    },
]

train_data = Dataset.from_list(examples)

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
)

args = SFTConfig(
    output_dir="outputs/plain-language-lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=1,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model_id,
    args=args,
    train_dataset=train_data,
    processing_class=tokenizer,
    peft_config=lora,
)

trainer.train()
trainer.model.save_pretrained("outputs/plain-language-lora/adapter")
tokenizer.save_pretrained("outputs/plain-language-lora/adapter")
~~~

The two examples make the code executable, not the resulting adapter useful. A real training run needs sufficient licensed data, a held-out release set, secure handling, hardware-appropriate precision, and a documented baseline.

The official PEFT interface creates a LoraConfig, wraps the base model with trainable adapters, and can report which parameters are trainable [5]. Use that report as an assertion, not decoration:

~~~python
trainer.model.print_trainable_parameters()

matched = [
    name for name, parameter in trainer.model.named_parameters()
    if parameter.requires_grad
]
assert matched, "No trainable adapter parameters were found"
~~~

For QLoRA, load the base model with an explicit 4-bit BitsAndBytesConfig, prepare it for k-bit training, and apply LoRA. Keep that configuration separate from the plain-LoRA baseline so memory savings and quality changes can be attributed.

## Connecting the workflow to FinPlainLM

[FinPlainLM](https://huggingface.co/datasets/Akshat4112/finplainlm-dpo-dataset) is an example of separating the two learning objectives. Its repository exposes SFT data for financial plain-language rewriting and preference data with chosen and rejected responses [6].

For this kind of task:

- SFT teaches the mapping from technical financial language to an intended plain-language response;
- DPO can refine preferences such as clarity, faithfulness, brevity, and preservation of material details; and
- a terminology or numerical-consistency evaluator can detect simplifications that change the underlying meaning.

The dataset link is evidence that the artefact exists, not evidence that a particular model trained on it meets a production target. Report dataset provenance, licences, splits, filtering, deduplication, model configuration, and evaluation results before making a performance claim.

## Design the dataset around failure slices

A fine-tuning dataset should not be a large undifferentiated collection. Attach metadata that supports analysis:

- task and document type;
- language and locale;
- input and response length;
- risk level;
- presence of numbers, tables, or specialist terminology;
- desired behaviour, such as answer, abstain, or refuse; and
- source and licence.

For financial plain-language rewriting, useful slices include:

| Slice | Failure to test |
|---|---|
| Interest rates and percentages | Numeric value or unit changes |
| Legal obligations | Modality changes from “must” to “may” |
| Long definitions | Important conditions are omitted |
| Tables and lists | Structure is flattened incorrectly |
| Specialist terms | Simplification becomes factually wrong |
| Already-simple text | Model rewrites unnecessarily |

Deduplicate semantically similar records across train and evaluation sets. If synthetic data is used, sample and review it before training; generating examples with a strong model does not make their labels correct.

## Evaluate the intervention, not only training loss

Training loss measures fit to training tokens. It does not answer whether the adapted model is better for users.

Compare at least:

1. the base model with the production prompt;
2. the base model plus retrieval or tools, where relevant;
3. the LoRA-adapted model;
4. the QLoRA variant, if memory is the reason for using it; and
5. full fine-tuning only when its additional cost is justified.

Keep prompts and decoding settings fixed during a controlled comparison. Evaluate:

- task accuracy or rubric scores by slice;
- format and schema validity;
- factual and numerical consistency;
- refusal and abstention behaviour;
- regressions on general capabilities;
- latency and throughput with adapter loading or merging;
- accelerator memory during training and serving; and
- cost per successful task.

Repeated runs matter when decoding or training is stochastic. Report seeds, model and dataset revisions, the number of evaluated examples, and uncertainty around differences.

### Check for regressions

Fine-tuning can improve the target behaviour while damaging unrelated behaviour or safety constraints. Preserve a regression suite for:

- general instruction following;
- multilingual behaviour needed by the application;
- safety and access-control prompts;
- long-context use;
- structured outputs;
- tool calls; and
- known production failures.

Do not merge an adapter into the base weights until the separately loaded adapter has passed the same release checks. Merging can simplify serving, but it changes rollback and artefact-management choices.

## Operational considerations

LoRA reduces adaptation cost, but production still needs model governance.

- Version the base model, tokenizer, adapter, training code, data manifest, and chat template together.
- Verify that the base model and dataset licences permit the intended use.
- Record the exact target modules, rank, scaling, dropout, precision, seed, optimiser, and checkpoints.
- Scan training data for personal, confidential, or restricted information.
- Keep adapters isolated by tenant or purpose when combining them could leak behaviour or data.
- Test adapter switching and batching in the actual serving runtime.
- Monitor post-release quality by the same slices used offline.

Adapter composition is not guaranteed to combine behaviours cleanly. Evaluate every composition as a new model configuration.

## Common mistakes

| Mistake | Why it fails |
|---|---|
| Training on documents to “add knowledge” | Facts become difficult to update, cite, or remove |
| Choosing LoRA before establishing a baseline | There is no evidence that weight adaptation is needed |
| Treating QLoRA as a package | It confuses a training method with its implementations |
| Reporting only trainable-parameter percentage | It says nothing about task quality or total runtime memory |
| Copying target-module names between architectures | The adapter may attach to unintended or nonexistent layers |
| Evaluating on training-style examples only | Memorisation and template matching look like generalisation |
| Ignoring rejected-response quality in DPO | Preference labels become noisy or trivial |
| Saving only adapter weights | The result is not reproducible without its base revision and tokenizer |

## Practical takeaway

Fine-tuning should follow diagnosis. Use prompting for instructions that the base model can already follow, RAG for inspectable changing knowledge, tools for authoritative operations, SFT for demonstrated behaviour, and preference optimisation for defensible comparisons between responses.

When weight adaptation is justified, LoRA offers a parameter-efficient representation:

\[
W' = W_0 + \frac{\alpha}{r}BA.
\]

Its value is not that fine-tuning becomes universally cheap or easy. Its value is that a measured behavioural change can be trained, versioned, evaluated, and deployed as a smaller artefact while the base model remains frozen.

## References

1. Hu, E. J. et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685). arXiv:2106.09685.
2. Dettmers, T. et al. (2023). [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314). arXiv:2305.14314.
3. Rafailov, R. et al. (2023). [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290). arXiv:2305.18290.
4. Ouyang, L. et al. (2022). [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155). arXiv:2203.02155.
5. Hugging Face. [*PEFT LoRA documentation*](https://huggingface.co/docs/peft/package_reference/lora). Accessed 10 September 2026.
6. Gupta, A. [*FinPlainLM DPO Dataset*](https://huggingface.co/datasets/Akshat4112/finplainlm-dpo-dataset). Hugging Face Datasets.
