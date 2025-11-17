# Mistral-7B Business & Product Assistant (QLoRA Fine-Tuning)

This repository contains a small but complete experiment in fine-tuning a
4-bit quantized **Mistral-7B-Instruct** language model on a subset of the
**Databricks Dolly 15k** dataset. The objective is to turn the base model
into a lightweight *Business & Product Development Assistant* that can
suggest realistic AI use cases in industrial and corporate settings.

The project is intentionally compact: one notebook, one dataset, and a
modern fine-tuning stack combining **Unsloth**, **TRL**, and **PEFT**.
It is designed as a learning exercise and as a concrete portfolio piece
for practical LLM work.



## Motivation

Large language models are increasingly used in product development and
enterprise workflows (ideation, documentation, decision support, internal
tools). However, most public models are trained for broad, general-purpose
chatting.

This experiment explores how far a modest amount of supervised fine-tuning
can push a strong open model (Mistral-7B) towards a more focused role:
an assistant that:

- understands business- and product-related instructions,
- proposes concrete AI applications in companies,
- communicates in a concise, structured way.

The setting is close to real-world use cases in industrial companies and
is also relevant for internships and practical work in applied AI.



## Technical Overview

**Base model**

- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- 4-bit quantized Mistral-7B-Instruct, prepared for QLoRA-style training

**Fine-tuning method**

- **QLoRA**: 4-bit base model + low-rank LoRA adapters
- Only a small fraction of parameters are trainable (parameter-efficient fine-tuning)

**Core libraries**

- [Unsloth](https://github.com/unslothai/unsloth) – fast, memory-efficient LLM fine-tuning
- [TRL](https://github.com/huggingface/trl) – `SFTTrainer` for supervised fine-tuning
- [PEFT](https://github.com/huggingface/peft) – parameter-efficient fine-tuning (LoRA)
- Hugging Face:
  - `transformers`
  - `datasets`
  - `accelerate`
  - `bitsandbytes` (4-bit quantization)

**Environment**

- Google Colab GPU (e.g., T4)
- Python 3
- No special infrastructure beyond a standard Colab session



## Dataset

The model is fine-tuned on a subset of the **Databricks Dolly 15k** dataset.

- Source: `databricks/databricks-dolly-15k` on Hugging Face
- Type: human-written instruction–response pairs
- Content: a broad mix of Q&A, brainstorming, classification, and
  summarization tasks, many of them business- or work-related
- For this experiment, a shuffled subset of 1,000 examples is used

Each example is converted into a simple *chat-style* format:

- A system message that defines the assistant role (“assistant for product
  development, business decisions, AI in companies”),
- A user message built from Dolly’s `instruction` (+ optional `context`),
- An assistant message from Dolly’s `response`.

These triplets are then transformed into the model’s chat template using
Unsloth’s `apply_chat_template`.



## Training Setup

The full training workflow is implemented in:

- `mistral-business-assistant-lora.ipynb`

The notebook performs the following steps:

1. Load Dolly 15k and select a subset of 1,000 examples.
2. Convert each example into a chat-style `messages` structure.
3. Load the 4-bit quantized Mistral-7B-Instruct model via `FastLanguageModel`.
4. Attach LoRA adapters with Unsloth (QLoRA configuration).
5. Use TRL’s `SFTTrainer` to run supervised fine-tuning for one epoch.
6. Save:
   - a LoRA-only checkpoint, and
   - a merged 16-bit model checkpoint.

The effective number of trainable parameters is printed to highlight the
parameter-efficiency of the QLoRA setup.

Hyperparameters are intentionally simple (single epoch, small subset of
data) to keep the experiment reproducible in a single Colab session.



## Evaluation

The notebook includes a small qualitative evaluation based on a few
hand-crafted prompts that reflect likely usage in an industrial context.
For example:

```text
You are an AI consultant in a large industrial company.
Suggest three concrete ways to use machine learning in predictive
maintenance for production lines.The same prompts can be used to compare:

the base model (before fine-tuning), and

the fine-tuned model.

In the notebook, both versions are queried, and their responses can be
compared side by side. The goal is not to reach state-of-the-art
benchmarks, but to illustrate how even a relatively small amount of
task-specific fine-tuning changes the behaviour of the model.

A simple training loss curve is also plotted to inspect convergence
during fine-tuning.
```
