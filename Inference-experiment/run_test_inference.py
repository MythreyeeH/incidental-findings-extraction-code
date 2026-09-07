"""
Qwen2.5-0.5B base (Q8_0) + swappable LoRA adapters — llama.cpp inference + benchmarking
=========================================================================================
One base GGUF, two adapter GGUFs applied on top at load time:
  • Base:      models/qwen2.5-0.5b-instruct-q8_0.gguf
  • Thoracic:  models/thoracic_adapter_q8_0.gguf   → 1-shot eval
  • Abdomen:   models/abdomen_adapter_q8_0.gguf    → 3-shot eval

Note on llama-cpp-python's LoRA API: the adapter is attached via the `lora_path`
argument when constructing `Llama(...)`. There's no supported public method to
hot-swap the adapter on an already-loaded `Llama` instance without rebuilding it,
so we reload (base weights are mmap'd, and the model is only 0.5B, so this is fast
and cheap) rather than keep one instance alive across both adapters.

Metrics: TTFT, Latency, Time-per-token, Peak RAM
"""

import json, re, os, time, statistics, random, gc
from pathlib import Path

import psutil
import pandas as pd
from tqdm import tqdm
import llama_cpp
from llama_cpp import Llama

# ── Config ─────────────────────────────────────────────────────────────────
BASE_GGUF          = "models/qwen2.5-0.5b-instruct-q8_0.gguf"
THORACIC_ADAPTER   = "models/qlora_thoracic_adapter.gguf"
ABDOMEN_ADAPTER    = "models/qlora_abdomen_adapter.gguf"
LORA_SCALE         = 1.0

N_CTX        = 4096
N_THREADS    = os.cpu_count()
MAX_TOKENS   = 256
N_GPU_LAYERS = 0

THORACIC_ANNOTATED   = "Dataset/TestDataset_Thoractic_CT/test_dataset_thoracic_annotated.json"
THORACIC_UNANNOTATED = "Dataset/TestDataset_Thoractic_CT/test_dataset_thoracic_unannotated.json"
ABDOMEN_TEST         = "Dataset/test_dataset_abdomen_final/test_records_abdomenCT.jsonl"
THORACIC_FEW_SHOT    = "Dataset/qwen_fewshot_thoractic/thoractic_few_shot_examples.json"
ABDOMEN_FEW_SHOT     = "Dataset/fewshot_abdomen/few_shot_abd_ct.json"

# ── RAM helper ─────────────────────────────────────────────────────────────
def peak_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# ── Model loader ───────────────────────────────────────────────────────────
def load_base() -> Llama:
    """Load the base model once, with no adapter attached."""
    print(f"Loading base model {BASE_GGUF} …")
    llm = Llama(
        model_path=BASE_GGUF,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print("Base model loaded.\n")
    return llm

# ── In-place LoRA attach/detach (low-level C API) ───────────────────────────
# The high-level Llama class only supports binding a LoRA at construction time
# (via lora_path). To swap adapters on an already-loaded model without
# reloading the base weights, we drop down to llama.cpp's C API, which does
# support this: llama_adapter_lora_init / llama_set_adapter_lora /
# llama_rm_adapter_lora / llama_adapter_lora_free.
#
# NOTE: llm._model.model and llm._ctx.ctx are private internals of
# llama-cpp-python's Llama class (not public API). They've been stable across
# recent versions, but if attach/detach starts raising AttributeError after
# upgrading llama-cpp-python, check llama_cpp/llama.py in your installed
# package for the current attribute names.

def _attach_adapter(llm: Llama, adapter_path: str, scale: float = LORA_SCALE):
    adapter = llama_cpp.llama_adapter_lora_init(
        llm._model.model, adapter_path.encode("utf-8")
    )
    if not adapter:
        raise RuntimeError(f"Failed to init LoRA adapter from {adapter_path}")

    ret = llama_cpp.llama_set_adapter_lora(llm._ctx.ctx, adapter, scale)
    if ret != 0:
        raise RuntimeError(f"Failed to apply LoRA adapter from {adapter_path} (ret={ret})")

    return adapter

def _detach_adapter(llm: Llama, adapter):
    llama_cpp.llama_rm_adapter_lora(llm._ctx.ctx, adapter)   # detach from context
    llama_cpp.llama_adapter_lora_free(adapter)                # free adapter weights

def attach_thoracic_weights(llm: Llama):
    print(f"Attaching thoracic adapter ({THORACIC_ADAPTER}) …")
    adapter = _attach_adapter(llm, THORACIC_ADAPTER)
    print("Thoracic adapter attached.\n")
    return adapter

def attach_abdomen_weights(llm: Llama):
    print(f"Attaching abdomen adapter ({ABDOMEN_ADAPTER}) …")
    adapter = _attach_adapter(llm, ABDOMEN_ADAPTER)
    print("Abdomen adapter attached.\n")
    return adapter

def detach_weights(llm: Llama, adapter, label: str = ""):
    _detach_adapter(llm, adapter)
    print(f"{label + ' ' if label else ''}adapter detached.\n")

# ── Data loaders ───────────────────────────────────────────────────────────
def load_thoracic():
    with open(THORACIC_ANNOTATED) as f:
        annotated = json.load(f)
    with open(THORACIC_UNANNOTATED) as f:
        unannotated = json.load(f)
    lookup = {r["report_id"]: r["annotation"] for r in annotated["reports"]}
    records = []
    for r in unannotated["reports"]:
        rid = r["report_id"]
        if rid in lookup:
            records.append({"report_id": rid, "free_text": r["free_text"], "gold": lookup[rid]})
    print(f"Thoracic test: {len(records)}")
    return records

def load_abdomen():
    records = []
    with open(ABDOMEN_TEST) as f:
        for i, line in enumerate(f):
            rec  = json.loads(line)
            msgs = rec["messages"]
            records.append({
                "report_id": f"abd_{i}",
                "free_text": msgs[1]["content"].replace("Report:\n", "").strip(),
                "gold":      json.loads(msgs[2]["content"]),
            })
    print(f"Abdomen test: {len(records)}")
    return records

def normalize_few_shot(pool):
    return [{
        "free_text": ex["free_text"],
        "gold": {"contains_IF": ex["contains_IF"], "incidental_sentences": ex["incidental_sentences"]},
    } for ex in pool]

def load_few_shot():
    with open(THORACIC_FEW_SHOT) as f:
        th = normalize_few_shot(json.load(f))
    with open(ABDOMEN_FEW_SHOT) as f:
        ab = normalize_few_shot(json.load(f))
    print(f"Thoracic few-shot: {len(th)}, Abdomen few-shot: {len(ab)}")
    return th, ab

# ── Prompt builder ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a clinical assistant. Extract the exact sentence(s) containing "
    "incidental findings from the report. If none are present, return an empty list. "
    "Always respond with valid JSON in the format: "
    '{"contains_IF": true/false, "incidental_sentences": []}.'
)

def build_prompt(few_shot_pool, n_shot, free_text) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot_pool[:n_shot]:
        messages.append({"role": "user",      "content": f"Report:\n{ex['free_text']}"})
        messages.append({"role": "assistant", "content": json.dumps({
            "contains_IF":          ex["gold"]["contains_IF"],
            "incidental_sentences": ex["gold"]["incidental_sentences"],
        })})
    messages.append({"role": "user", "content": f"Report:\n{free_text}"})

    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)

# ── Output parser ──────────────────────────────────────────────────────────
def parse_output(text):
    if not text:
        return None
    for candidate in reversed(re.findall(r'\{[^{}]*\}', text, re.DOTALL)):
        try:
            parsed = json.loads(candidate)
            if "incidental_sentences" in parsed:
                return parsed
        except Exception:
            continue
    return None

# ── Inference + metric collection ──────────────────────────────────────────
def run_inference(llm, test_records, few_shot_pool, n_shot, desc=""):
    results, ttfts, latencies, tpts, ram_samples = [], [], [], [], []

    for record in tqdm(test_records, desc=f"{desc} {n_shot}-shot"):
        prompt = build_prompt(few_shot_pool, n_shot, record["free_text"])

        ram_before       = peak_ram_mb()
        t_start          = time.perf_counter()
        first_token_time = None
        generated_text   = ""
        n_tokens_gen     = 0

        stream = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        for chunk in stream:
            token_str = chunk["choices"][0]["text"]
            if first_token_time is None and token_str:
                first_token_time = time.perf_counter()
            generated_text += token_str
            n_tokens_gen   += 1

        t_end   = time.perf_counter()
        ram_after = peak_ram_mb()

        ttft    = (first_token_time - t_start) if first_token_time else float("nan")
        latency = t_end - t_start
        tpt     = latency / n_tokens_gen if n_tokens_gen > 0 else float("nan")

        ttfts.append(ttft)
        latencies.append(latency)
        tpts.append(tpt)
        ram_samples.append(max(ram_before, ram_after))

        parsed = parse_output(generated_text.strip())
        results.append({
            "report_id":        record["report_id"],
            "gold_contains_IF": record["gold"]["contains_IF"],
            "gold_sentences":   record["gold"]["incidental_sentences"],
            "pred_sentences":   parsed.get("incidental_sentences") if parsed else None,
            "raw_output":       generated_text.strip(),
            "parse_failed":     parsed is None,
            "ttft_s":           round(ttft,     4),
            "latency_s":        round(latency,  4),
            "tpt_s":            round(tpt,      4),
            "n_tokens_gen":     n_tokens_gen,
            "peak_ram_mb":      round(ram_after, 1),
        })

    metrics = {
        "ttft_mean_s":    round(statistics.mean(ttfts),      4),
        "ttft_std_s":     round(statistics.stdev(ttfts),     4) if len(ttfts) > 1 else 0.0,
        "latency_mean_s": round(statistics.mean(latencies),  4),
        "latency_std_s":  round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "tpt_mean_s":     round(statistics.mean(tpts),       4),
        "tpt_std_s":      round(statistics.stdev(tpts),      4) if len(tpts) > 1 else 0.0,
        "peak_ram_mb":    round(max(ram_samples),            1),
    }
    return results, metrics

# ── Evaluation ─────────────────────────────────────────────────────────────
def run_evaluation(results, metrics, label=""):
    parse_failures = sum(r["parse_failed"] for r in results)
    valid          = [r for r in results if not r["parse_failed"]]
    total_tp = total_fp = total_fn = 0
    neg_scores, pos_scores = [], []

    for r in valid:
        gold_set = set(s.strip().lower() for s in r["gold_sentences"])
        pred_set = set(s.strip().lower() for s in (r["pred_sentences"] or []))
        if not gold_set:
            neg_scores.append(1.0 if not pred_set else 0.0)
        else:
            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            total_tp += tp; total_fp += fp; total_fn += fn
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            pos_scores.append(2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0)

    avg_neg     = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
    avg_pos     = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    n_neg, n_pos = len(neg_scores), len(pos_scores)
    macro_f1    = (avg_neg + avg_pos) / 2 if (n_neg > 0 and n_pos > 0) else (avg_neg or avg_pos)
    weighted_f1 = (n_neg * avg_neg + n_pos * avg_pos) / (n_neg + n_pos) if (n_neg + n_pos) > 0 else 0.0
    micro_p     = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r     = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1    = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"=== {label} ===")
    print(f"{'='*55}")
    print(f"Parse failures:      {parse_failures}/{len(results)}")
    print(f"Sentence Macro F1:   {macro_f1:.4f}")
    print(f"Sentence Weighted:   {weighted_f1:.4f}")
    print(f"Sentence Micro F1:   {micro_f1:.4f}")
    print(f"--- Latency Metrics ---")
    print(f"TTFT:                {metrics['ttft_mean_s']:.4f}s  ± {metrics['ttft_std_s']:.4f}s")
    print(f"Latency (total gen): {metrics['latency_mean_s']:.4f}s ± {metrics['latency_std_s']:.4f}s")
    print(f"Time-per-token:      {metrics['tpt_mean_s']:.4f}s  ± {metrics['tpt_std_s']:.4f}s")
    print(f"Peak RAM:            {metrics['peak_ram_mb']} MB")

def stratified_sample(records, n, seed=42):
    """Stratified sample preserving the proportion of contains_IF positive/negative records."""
    pos = [r for r in records if r["gold"]["contains_IF"]]
    neg = [r for r in records if not r["gold"]["contains_IF"]]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    total = len(records)
    n_pos = round(n * len(pos) / total) if total > 0 else 0
    n_neg = n - n_pos

    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))
    shortfall = n - (n_pos + n_neg)
    if shortfall > 0:
        if len(pos) - n_pos >= shortfall:
            n_pos += shortfall
        else:
            n_neg += shortfall

    sample = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(sample)
    return sample

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    thoracic_few_shot, abdomen_few_shot = load_few_shot()

    thoracic_all = load_thoracic()
    abdomen_all  = load_abdomen()

    thoracic_test = stratified_sample(thoracic_all, 50)
    abdomen_test  = stratified_sample(abdomen_all, 50)

    print(f"Thoracic stratified sample: {sum(r['gold']['contains_IF'] for r in thoracic_test)} pos / {len(thoracic_test) - sum(r['gold']['contains_IF'] for r in thoracic_test)} neg")
    print(f"Abdomen stratified sample: {sum(r['gold']['contains_IF'] for r in abdomen_test)} pos / {len(abdomen_test) - sum(r['gold']['contains_IF'] for r in abdomen_test)} neg")

    # ── Load base model once ─────────────────────────────────────────────
    llm = load_base()

    # ── Thoracic: attach thoracic adapter in place, 1-shot ──────────────────
    thoracic_adapter = attach_thoracic_weights(llm)

    for n_shot in [1,]:
        results, metrics = run_inference(llm, thoracic_test, thoracic_few_shot, n_shot, desc="Thoracic")
        run_evaluation(results, metrics, label=f"Thoracic {n_shot}-Shot (Base+LoRA) 0.5B Q8_0")
        pd.DataFrame(results).to_csv(f"thoracic_{n_shot}shot_0.5b_lora_llamacpp.csv", index=False)

    detach_weights(llm, thoracic_adapter, label="Thoracic")

    # ── Abdomen: attach abdomen adapter in place, 3-shot ─────────────────────
    abdomen_adapter = attach_abdomen_weights(llm)

    for n_shot in [3,]:
        results, metrics = run_inference(llm, abdomen_test, abdomen_few_shot, n_shot, desc="Abdomen")
        run_evaluation(results, metrics, label=f"Abdomen {n_shot}-Shot (Base+LoRA) 0.5B Q8_0")
        pd.DataFrame(results).to_csv(f"abdomen_{n_shot}shot_0.5b_lora_llamacpp.csv", index=False)

    detach_weights(llm, abdomen_adapter, label="Abdomen")

    llm.close()
    del llm
    gc.collect()

    print("\nDone. CSVs saved with per-record metrics columns.")