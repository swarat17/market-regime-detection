# PEFT Regime Benchmark

![CI](https://github.com/YOUR_USERNAME/peft-regime-benchmark/actions/workflows/ci.yml/badge.svg)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME/peft-regime-benchmark)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Project-orange)](https://wandb.ai/YOUR_USERNAME/peft-regime-benchmark)

A rigorous benchmark comparing **LoRA, QLoRA, Prefix Tuning, and (IA)³** fine-tuning methods on **market regime detection** — classifying financial text (Fed statements, earnings calls, macro news) as **Bull**, **Bear**, or **Volatile**.

The key novel contribution is the **Regime Confidence Score**: a custom metric that penalizes high-confidence wrong predictions with financial severity weighting (Bull↔Bear errors penalized 2× more than adjacent errors).

---

## The Problem

Misclassifying market regimes has asymmetric consequences. Calling a Bear market **Bull** (opposite direction) is catastrophically more dangerous than calling it **Volatile** (adjacent). Standard accuracy and F1 metrics treat all errors equally — our Regime Confidence Score reflects the real financial cost of confident mistakes.

---

## Architecture

### PEFT Methods Compared

| Method | Key Config | Trainable Params |
|---|---|---|
| LoRA | r=16, α=32, fp4 quant | ~TBD |
| QLoRA | r=16, α=32, nf4 + double quant | ~TBD |
| Prefix Tuning | 10 virtual tokens | ~TBD |
| (IA)³ | learned scaling vectors | ~TBD |

**Base model:** Llama-3.2-3B (4-bit quantized, GTX 1650 / 4GB VRAM)
**Dataset:** FinancialPhraseBank → regime-relabeled (Bull/Bear/Volatile)

### Regime Confidence Score

For each wrong prediction:
```
penalty = confidence × severity_weight
severity: Bull↔Bear = 2.0×,  Bull↔Volatile or Bear↔Volatile = 1.0×
score = 1.0 - Σ(penalties) / (N × 2.0)   ∈ [0, 1]
```

---

## Benchmark Results

> Results will be filled in after training completes.

| Method | Base Model | Accuracy | F1 Macro | Bull F1 | Bear F1 | Volatile F1 | Regime Confidence Score | Trainable Params |
|---|---|---|---|---|---|---|---|---|
| LoRA | Llama-3.2-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| QLoRA | Llama-3.2-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Prefix Tuning | Llama-3.2-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| (IA)³ | Llama-3.2-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## LoRA Rank Sensitivity

> Plot will be generated after rank sweep completes.

![Rank Sensitivity](results/plots/rank_sensitivity.png)

| LoRA Rank (r) | F1 Macro | Regime Confidence Score | Trainable Params | Training Time (s) |
|---|---|---|---|---|
| 4 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD | TBD |

---

## Key Findings

> To be filled in after training completes.

- **TBD** — best performing method and its Regime Confidence Score
- **TBD** — LoRA rank sweet spot (accuracy/compute tradeoff)
- **TBD** — which method handles Bear misclassification best

---

## Reproduction

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/peft-regime-benchmark.git
cd peft-regime-benchmark

# 2. Install PyTorch with CUDA (do this first)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Authenticate
huggingface-cli login   # required for Llama-3.2-3B (gated model)
wandb login

# 5. Train a single method
python scripts/train.py --method lora --base-model llama3 --seed 42

# 6. Evaluate all checkpoints
python scripts/evaluate.py --auto-discover

# 7. Run LoRA rank sensitivity sweep
python scripts/rank_sensitivity.py
```

Runs under **~$5 compute cost** using 4-bit QLoRA on a consumer GPU.

---

## Project Structure

```
├── configs/           # YAML configs for each PEFT method
├── src/
│   ├── data/          # FPB loader, regime labeling, stratified splits
│   ├── models/        # Base model loader, PEFT factory
│   ├── training/      # Trainer, W&B callbacks
│   └── evaluation/    # Metrics, evaluator, inference
├── scripts/           # train.py, evaluate.py, rank_sensitivity.py
├── frontend/          # Gradio app (Phase 6)
└── tests/             # 19 unit tests, integration tests
```

---

## Stack

HuggingFace PEFT · bitsandbytes · Transformers · Weights & Biases · Llama-3.2-3B · Gradio · Hugging Face Spaces
