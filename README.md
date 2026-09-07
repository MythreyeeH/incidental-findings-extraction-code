# Incidental Findings Extraction from Radiology Reports

This repository contains code and experiments for extracting **incidental findings** from radiology reports (thoracic and abdominal CT) using **Qwen2.5-0.5B-Instruct**, fine-tuned with knowledge distillation and QLoRA.

The project is organized into three self-contained experiment folders, each with its own README covering setup and usage in detail.

---

## Repository Structure
```
.
├── Full-pipeline-experiment/ # Full 3-stage training pipeline: KD → QLoRA → self-consistency voting
├── Human-in-the-loop-experiment/ # Radiologist correction memory on top of the fine-tuned models
├── Inference-experiment/ # CPU inference using llama.cpp + GGUF quantized models
└── README.md
```

---

## Experiments

### 1. [Full Pipeline Experiment](./Full-pipeline-experiment/README.md)
Trains organ-specific (thoracic and abdominal) extraction models via a three-stage pipeline: knowledge distillation from a Qwen2.5-7B-Instruct teacher, QLoRA fine-tuning, and self-consistency voting across stochastic samples. Includes the notebooks, datasets, and stage ordering needed to reproduce training end-to-end.

### 2. [Human-in-the-Loop Experiment](./Human-in-the-loop-experiment/README.md)
Adds an updatable correction memory on top of the fine-tuned extraction models, letting radiologists correct model output at the sentence level. Corrections are retrieved via embedding similarity (Bio_ClinicalBERT) and applied to future inference without retraining. Includes self-consistency and pair-generalization evaluation.

### 3. [Inference Experiment](./Inference-experiment/README.md)
CPU inference pipeline using `llama.cpp` and GGUF-quantized models. Covers converting the base model and QLoRA adapters to GGUF, quantizing to Q8_0, and running inference with the pre-trained thoracic and abdominal adapters.

---

## Getting Started

Each folder is independent and contains its own README with:

- Repository structure specific to that experiment
- Setup instructions (Kaggle notebook setup, data upload, path configuration)
- How to run the code
- Model and methodology details

Navigate to the experiment you're interested in and follow its README to reproduce the results.

---

## Base Model

- **Qwen2.5-0.5B-Instruct** — fine-tuned across all experiments via knowledge distillation and QLoRA
- **Qwen2.5-7B-Instruct** — used as the teacher model for knowledge distillation (Full Pipeline Experiment)