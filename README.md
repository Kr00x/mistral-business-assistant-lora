# Mistral-7B Business & Product Assistant (QLoRA Fine-Tuning)

This repository contains a focused experiment in fine-tuning a 4-bit quantized
**Mistral-7B-Instruct** language model on a subset of the **Databricks Dolly
15k** dataset. The goal is to adapt a strong open model into a lightweight
*Business & Product Development Assistant* that can suggest realistic AI use
cases in industrial and corporate settings.

The project is intentionally compact: one notebook, one dataset, and a modern
fine-tuning stack built around **Unsloth**, **TRL**, and **PEFT**. It is meant
both as a learning exercise and as a concrete portfolio piece that shows how
to work with contemporary LLM tooling in a way that is reproducible on a
standard Colab GPU.

## Motivation

Large language models are becoming part of everyday product development and
enterprise workflows, from ideation and documentation to decision support and
internal tools. Most public models, however, are trained for broad,
general-purpose chat and are not specialized for the kind of questions that
come up in industrial companies or product teams.

This project explores how far a modest amount of supervised fine-tuning can
push a general model like Mistral-7B toward a more focused role: an assistant
that understands business- and product-related instructions, proposes concrete
AI applications in companies, and structures its answers in a way that would
actually be useful to engineers, product managers, or decision-makers. The
setting is deliberately close to real-world use cases in industrial
environments and is directly relevant for internships and applied AI work.

## Model and Methods

The base model is `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`, a 4-bit
quantized variant of Mistral-7B-Instruct prepared for QLoRA-style training.
Instead of updating all parameters of the model, the experiment uses
**QLoRA**, where the 4-bit base model remains frozen and low-rank LoRA
adapters are trained on top. This parameter-efficient approach keeps memory
requirements low and makes it possible to run the entire fine-tuning process
in a single Colab session.

The implementation relies on the following toolchain. Unsloth provides a
wrapper (`FastLanguageModel`) that loads the quantized model, attaches LoRA
adapters, and exposes utilities for merging or exporting the fine-tuned
weights. TRL contributes the `SFTTrainer`, which handles supervised
fine-tuning on text data with a reasonably simple configuration. PEFT
underpins the LoRA mechanism and allows only a small fraction of parameters to
be trainable. The stack is completed by the usual Hugging Face components:
`transformers` for the model and tokenizer, `datasets` for data handling,
`accelerate` for device management, and `bitsandbytes` for 4-bit
quantization.

The effective number of trainable parameters is printed in the notebook to
highlight the parameter-efficiency of this setup. The idea is to stay as close
as possible to realistic constraints: limited GPU memory, limited time, and
the need for a clear, inspectable pipeline.

## Dataset and Formatting

The model is fine-tuned on a small, shuffled subset of **Databricks Dolly
15k**, which is available on the Hugging Face Hub as
`databricks/databricks-dolly-15k`. Dolly consists of human-written
instruction–response pairs that cover a wide range of tasks, including
question answering, brainstorming, classification, and summarization. Many of
the instructions are rooted in business or work-related scenarios, which makes
it a natural choice for this kind of assistant.

For this experiment, a subset of 1,000 examples is sampled and converted into
a simple chat format. Each item becomes a sequence of three messages: a system
message that defines the role of the assistant (“assistant for product
development, business decisions, AI in companies”), a user message built from
Dolly’s `instruction` (and optional `context`), and an assistant message taken
from Dolly’s `response`. This `messages` structure is then passed through
Unsloth’s `apply_chat_template` to produce the actual serialized text that the
model consumes during training.

The result is a small but reasonably diverse corpus of business- and
product-oriented prompts paired with human answers, expressed in the same
chat-style format that the model will later use at inference time.

## Training Setup

All steps are implemented in the notebook `mistral-business-assistant-lora.ipynb`.

The workflow starts by loading the Dolly subset and building the
chat-formatted `messages`. The 4-bit quantized Mistral-7B-Instruct model is
then loaded via `FastLanguageModel`, and LoRA adapters are attached using
Unsloth’s QLoRA configuration. After applying the appropriate chat template
to the data, TRL’s `SFTTrainer` is used to run supervised fine-tuning for a
single epoch.

The hyperparameters are intentionally simple: one epoch, a small subset of
data, and a batch configuration chosen so that the run remains stable on a
standard Colab GPU (for example, a T4). At the end of training, the notebook
saves both a LoRA-only checkpoint and a merged 16-bit model checkpoint. This
allows either further adapter-based experimentation or direct loading of the
merged model for inference.

A minimal training loss curve is plotted to give a quick visual check that the
optimization behaves as expected and that there are no obvious issues with
divergence.

## Evaluation

Evaluation in this project is deliberately qualitative rather than focused on
formal benchmarks. The notebook defines a small set of prompts that resemble
questions an industrial company or product team might ask. One example is:

```text
You are an AI consultant in a large industrial company.
Suggest three concrete ways to use machine learning in predictive
maintenance for production lines.
The same prompts can be posed to the base model and to the fine-tuned model.
Their responses can then be compared side by side to see how the behaviour
changes after training. The emphasis is on whether the assistant proposes
concrete, plausible AI applications, whether it stays on topic, and whether
the answers are structured in a way that would be useful in a real setting.
The experiment does not aim for state-of-the-art performance and does not
claim to be production-ready. Instead, it is meant to illustrate how a
relatively small amount of task-specific supervised fine-tuning can already
change the character of a general-purpose model and move it closer to a
specialized business assistant.
```

## Usage

To reproduce the experiment, open the notebook in Google Colab, enable a GPU
in the runtime settings, install the required dependencies in the first cell,
and run the cells from top to bottom. With the current configuration, the
entire run fits into a single Colab session on a typical GPU.
Fine-tuned weights are not checked into this repository. They can be exported
from Colab as a LoRA adapter directory and as a merged 16-bit model directory,
and optionally uploaded to the Hugging Face Hub for reuse in other projects.

::contentReference[oaicite:0]{index=0}
