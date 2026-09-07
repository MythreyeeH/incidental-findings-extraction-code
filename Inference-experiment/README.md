# Incidental Findings Extraction using QLoRA

This repository contains the code and trained QLoRA adapters for extracting incidental findings from radiology reports using **Qwen2.5-0.5B-Instruct**.

Two separate adapters are provided:

- **Thoracic CT incidental findings**
- **Abdominal CT incidental findings**

The repository also includes a CPU inference script using `llama.cpp` and GGUF models.

---

## Repository Structure

```
.
├── Abdomen_qlora_weights/          # Trained abdomen QLoRA adapter
├── Thoracic_qlora_weights/         # Trained thoracic QLoRA adapter
├── Dataset/                        # Training and evaluation datasets
├── qLoRA-abdomen.ipynb             # Kaggle notebook for abdomen fine-tuning
├── qlora-thoracic.ipynb            # Kaggle notebook for thoracic fine-tuning
├── run_test_inference.py           # CPU inference script
└── .gitignore
```

---

# Training

The training notebooks were developed and executed on **Kaggle**.

## Setup

1. Create a new Kaggle notebook.
2. Upload the folders inside the `Dataset/` directory as Kaggle Dataset inputs.
3. Open either notebook depending on the task:

- `qlora-thoracic.ipynb`
- `qLoRA-abdomen.ipynb`

4. Run all notebook cells.

The notebooks perform QLoRA fine-tuning of **Qwen2.5-0.5B-Instruct** and generate the adapter weights contained in this repository.

---

# CPU Inference

Inference is performed using **llama.cpp** and GGUF models.

## 1. Clone llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

---

## 2. Install Python Dependencies

```bash
pip install huggingface_hub safetensors torch transformers peft accelerate llama-cpp-python psutil pandas tqdm
```

---

## 3. Download the Base Model

Download the Qwen2.5-0.5B-Instruct model.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct',
    local_dir='path/qwen_0.5b_base'
)
print('Done!')
"
```

---

## 4. Convert the Base Model to GGUF (F16)

```bash
python path/llama.cpp/convert_hf_to_gguf.py \
"path/qwen_0.5b_base" \
--outfile "path/models/qwen2.5-0.5b-instruct-f16.gguf" \
--outtype f16
```

---

## 5. Quantize the Model to Q8_0

```bash
python -c "
from llama_cpp import llama_model_quantize, llama_model_quantize_default_params

params = llama_model_quantize_default_params()
params.ftype = 7   # Q8_0

llama_model_quantize(
    b'path\\\\models\\\\qwen2.5-0.5b-instruct-f16.gguf',
    b'path\\\\models\\\\qwen2.5-0.5b-instruct-q8_0.gguf',
    params
)

print('Base model Q8_0 done!')
"
```

---

## 6. Convert the QLoRA Adapters to GGUF

### Thoracic Adapter

```bash
python path/llama.cpp/convert_lora_to_gguf.py \
"path/Thoracic_qlora_weights" \
--outfile "path/models/qlora_thoracic_adapter.gguf" \
--base-model-id Qwen/Qwen2.5-0.5B-Instruct
```

### Abdomen Adapter

```bash
python path/llama.cpp/convert_lora_to_gguf.py \
"path/Abdomen_qlora_weights" \
--outfile "path/models/qlora_abdomen_adapter.gguf" \
--base-model-id Qwen/Qwen2.5-0.5B-Instruct
```

---

## 7. Expected Directory Structure

Before running inference, your project should contain:

```
.
├── models/
│   ├── qwen2.5-0.5b-instruct-q8_0.gguf
│   ├── qlora_thoracic_adapter.gguf
│   └── qlora_abdomen_adapter.gguf
│
├── Dataset/
├── Thoracic_qlora_weights/
├── Abdomen_qlora_weights/
├── run_test_inference.py
└── ...
```

---

## 8. Run Inference

Execute the inference script in a CPU environment:

```bash
python run_test_inference.py
```

The script loads:

- Quantized Qwen2.5-0.5B-Instruct GGUF model
- Thoracic or Abdomen GGUF adapter
- Test reports
- Predicted incidental findings

---

# Model

**Base Model**

- Qwen2.5-0.5B-Instruct

**Fine-tuning**

- QLoRA
- GGUF conversion for CPU inference
- Q8_0 quantized base model
