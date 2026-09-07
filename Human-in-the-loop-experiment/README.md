# Human-in-the-Loop Memory Mechanism for Incidental Findings

This repository contains the code for a human-in-the-loop (HITL) correction memory built on top of the QLoRA-fine-tuned **Qwen2.5-0.5B-Instruct** incidental findings extraction models (thoracic and abdominal CT).

The mechanism lets radiologists correct model output on individual sentences, stores those corrections, and reuses them on future inference **without retraining the model**.

---

## Repository Structure
```
.
├── Data/
│   ├── Thoracic_data/      # Dataset for thoracic modality
│   └── Abdomen_data/       # Dataset for abdomen modality
├── Human-in-the-loop-experiment/thoracic-human-in-loop-logic.ipynb     # Kaggle notebook (modality = "thoracic")
├── Human-in-the-loop-experiment/abdomen-human-in-loop-logic.ipynb     # Kaggle notebook (modality = "abdomen")
└── .gitignore
```

Both notebooks are identical apart from a single `modality` variable in `main()`, which switches between the thoracic and abdomen pipelines (models, paths, and data).

---

## How It Works

### 1. Memory Representation
Corrections are stored at the sentence level. Each entry in `SentenceMemory` consists of:

- **trigger sentence** — the original finding sentence as it appears in the report
- **corrected sentence** — the radiologist-corrected replacement, or `None` if the sentence should be excluded entirely
- **embedding** — a NumPy array embedding of the trigger sentence, produced by a **Bio_ClinicalBERT** encoder

### 2. Retrieval
A candidate sentence is embedded with the same encoder and compared via cosine similarity against every stored trigger sentence. If the best match exceeds a similarity threshold, the stored correction is returned as a hit.

### 3. Override
Every sentence the extraction model flags as an incidental finding is checked against memory. On a match, the model's output for that sentence is replaced with the corrected version.

### 4. Evaluation Methodology
The mechanism is validated against two properties, tested independently:

1. **Self-consistency** — resending the exact same report after a correction should reliably return the corrected finding, not the original.
2. **Pair generalization** — a correction applied to one report's finding should transfer to a near-duplicate finding phrased differently in a separate report, via embedding-similarity matching rather than exact-text recall.

Results for both strategies, along with additional generalizability and threshold-sensitivity metrics, are reported in the Results section.

---

## Setup

1. Create a new Kaggle notebook.
2. Upload `Thoracic_data/` and `Abdomen_data/` as Kaggle Dataset inputs.
3. Open `hitl-thoracic.ipynb` or `hitl-abdomen.ipynb` depending on the modality.
4. Update the paths in the code to match your uploaded datasets.
5. Run all notebook cells.

---

## Model

**Base extraction model**

- Qwen2.5-0.5B-Instruct (after knowledge distillation + QLoRA fine-tuning)

**Correction encoder**

- Bio_ClinicalBERT (sentence embeddings for memory retrieval)