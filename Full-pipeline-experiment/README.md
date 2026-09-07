# Full Pipeline Experiment: Knowledge Distillation → QLoRA → Self-Consistency Voting

This repository contains the code for the full three-stage training pipeline for organ-specific incidental findings extraction models: **knowledge distillation (KD)**, **QLoRA fine-tuning**, and **self-consistency voting**. The pipeline is run separately for the thoracic and abdominal modalities.

---

## Repository Structure
```
Full-pipeline-experiment/
├── abdomen-data/
│ ├── CT_ABD_REPORTS_UNANNOTATED.json
│ ├── Findings_Extracted_WhitePapers.json
│ ├── test_records_abdomenCT.jsonl
│ └── train_records_abdomenCT.jsonl
├── abdomen-pipeline/
│ ├── abdomen-kd-1.ipynb
│ ├── abdomen-qlora-2.ipynb
│ └── abdomen-selfconsistencyvoting-3.ipynb
├── thoracic-data/
│ ├── few_shot_examples/
│ ├── test_dataset/
│ ├── train_dataset/
│ ├── unannotated_kd_dataset.json
│ └── white_paper_guidelines.json
├── thoracic-pipeline/
│ ├── thoracic-reverse-experiment-ensemble-....ipynb
│ ├── thoracic-reverse-experiment-kd-1.ipynb
│ └── thoracic-reverse-experiment-qlora2.ipynb
└── README.md
```

---

## Pipeline Overview

Each organ-specific model is trained via a three-stage pipeline: knowledge distillation, QLoRA fine-tuning, and self-consistency voting.

### 1. Knowledge Distillation
A **Qwen2.5-7B-Instruct** teacher model annotates the report pool, retaining the top-k=10 output logits at each generation step. The **Qwen2.5-0.5B-Instruct** student is trained on these reports using a hard-label focal loss on the annotated JSON output plus a KL-divergence term against the teacher's top-k logit distribution, transferring the teacher's decision behavior into the smaller model.

### 2. QLoRA Fine-Tuning
The knowledge-distilled model is further fine-tuned using QLoRA (4-bit NF4 quantization) on the annotated training reports, improving performance beyond the distillation stage alone.

### 3. Self-Consistency Voting
The KD+QLoRA model is evaluated using self-consistency voting: multiple stochastic samples are generated per report across a range of temperatures, and predictions are combined using a threshold vote across samples. The best-performing temperature/threshold configuration is selected on this basis.

### Data Sources & Splits

| Modality  | KD reports | QLoRA training reports | Test reports | Source |
|-----------|-----------|------------------------|---------------|--------|
| Thoracic  | 1,500     | 1,000                   | 100           | CT-RATE dataset |
| Abdominal | 1,500     | 1,000                   | 100           | 1,500 from RadGPT dataset; 1,000 train + 100 test supplied by a radiologist |

All reports were annotated using the strategy described in the paper. Performance is measured using sentence-level micro-F1, macro-F1, and weighted-F1, computed by fuzzy-matching predicted sentences against gold-annotated sentences at a similarity threshold of 0.85.

---

## How to Run

Each modality's pipeline (`abdomen-pipeline/` or `thoracic-pipeline/`) has three notebooks that must be run **in order**, on Kaggle:

1. **KD notebook** (`abdomen-kd-1.ipynb` / `thoracic-reverse-experiment-kd-1.ipynb`)
   - Load all data from the corresponding `abdomen-data/` or `thoracic-data/` folder as Kaggle Dataset inputs.
   - Run the notebook to produce the knowledge-distilled model weights.

2. **QLoRA notebook** (`abdomen-qlora-2.ipynb` / `thoracic-reverse-experiment-qlora2.ipynb`)
   - Load the KD output weights from step 1, along with the original data.
   - Run the notebook to produce the KD+QLoRA fine-tuned weights.

3. **Self-consistency voting notebook** (`abdomen-selfconsistencyvoting-3.ipynb` / `thoracic-reverse-experiment-ensemble-....ipynb`)
   - Load the KD+QLoRA weights from step 2, along with the original data.
   - Run the notebook to perform self-consistency voting and produce final evaluation metrics.

Update file paths in each notebook to match your uploaded Kaggle datasets before running.

---

## Model

**Teacher model**

- Qwen2.5-7B-Instruct

**Student model**

- Qwen2.5-0.5B-Instruct (knowledge distillation → QLoRA fine-tuning → self-consistency voting)