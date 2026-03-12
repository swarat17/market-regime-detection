# PEFT Methods Benchmark — Market Regime Detection
## CLAUDE.md — 7-Phase Build Plan

> Tell Claude Code which phase you're on at the start of each session. Each phase is
> 1–2 hours. Run tests before moving on.

## Implementation Status
- [ ] Pre-Phase: Environment Setup (requirements.txt, directory structure)
- [ ] Phase 1: Data Pipeline
- [ ] Phase 2: Model Loading & PEFT Factory
- [ ] Phase 3: Training Pipeline & W&B
- [ ] Phase 4: Custom Evaluation Metrics
- [ ] Phase 5: Benchmark Runner & LoRA Rank Sensitivity
- [ ] Phase 6: Gradio App & HF Spaces
- [ ] Phase 7: CI/CD & Documentation

## Hardware Constraints (GTX 1650, 4GB VRAM)
- **fp16 only** — Turing arch (sm_75), no bf16
- **batch_size=1** + gradient_accumulation_steps=16 + gradient_checkpointing=true
- **Mistral-7B**: coded but marked "requires >8GB VRAM"; default all runs to llama3
- **LoRA vs QLoRA**: Both quantized on this hardware; differ by quant_type (fp4 vs nf4) and double_quant
- **dataloader_num_workers=0** — Windows fork issue

## Key Design Decisions
- LoRA: `bnb_4bit_quant_type="fp4"`, `use_double_quant=False`
- QLoRA: `bnb_4bit_quant_type="nf4"`, `use_double_quant=True`
- Always `torch_dtype=torch.float16`
- `tokenizer.pad_token = tokenizer.eos_token` (Llama has no pad token)
- Use `pathlib.Path` everywhere for Windows compatibility
- Set `TOKENIZERS_PARALLELISM=false` in `src/utils/logger.py` at module level

---

## Project Overview

A rigorous benchmark comparing LoRA, QLoRA, Prefix Tuning, and (IA)³ fine-tuning methods
on market regime detection — classifying financial text (Fed statements, earnings calls,
macro news) as **Bull**, **Bear**, or **Volatile**. Two base models compared: Llama-3.2-3B
and Mistral-7B. All experiments tracked in Weights & Biases. Packaged as a reproducible
one-command repo with a companion blog post.

**The interesting angle:** A custom **Regime Confidence Score** (penalizes high-confidence
wrong predictions — especially dangerous in financial contexts) + a LoRA rank sensitivity
analysis (r = 4, 8, 16, 32, 64) showing the accuracy/compute tradeoff curve.

**Stack:** HuggingFace PEFT · bitsandbytes · Transformers · Weights & Biases ·
Llama-3.2-3B · Mistral-7B · Gradio · Hugging Face Spaces

**Datasets:**
- Primary: FinancialPhraseBank (sentiment → regime-relabeled)
- Augment with: Fed FOMC statements (public), earnings call transcripts subset from
  Kaggle, macro news headlines

**Final Repo Structure:**
```
peft-regime-benchmark/
├── CLAUDE.md
├── requirements.txt
├── pytest.ini
├── configs/
│   ├── lora.yaml
│   ├── qlora.yaml
│   ├── prefix_tuning.yaml
│   └── ia3.yaml
├── src/
│   ├── data/
│   │   ├── loader.py          # Phase 1
│   │   └── preprocessor.py    # Phase 1
│   ├── models/
│   │   ├── base_loader.py     # Phase 2
│   │   └── peft_factory.py    # Phase 2
│   ├── training/
│   │   ├── trainer.py         # Phase 3
│   │   └── callbacks.py       # Phase 3
│   ├── evaluation/
│   │   ├── metrics.py         # Phase 4
│   │   └── evaluator.py       # Phase 5
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── rank_sensitivity.py    # Phase 5
├── results/
│   ├── benchmark_results.csv
│   └── plots/
├── frontend/
│   └── app.py                 # Phase 6
└── tests/
    ├── unit/
    └── integration/
```

---

## Phase 1 — Data Pipeline
**Day 1 | ~1.5h**

### Goal
Build a reproducible data pipeline that loads, cleans, and regime-labels financial text.

### What to Build

**1. Regime labeling logic** (`src/data/preprocessor.py`)
Map existing sentiment/tone labels to three regime classes:
- **Bull**: positive outlook, growth language, beat expectations
- **Bear**: negative outlook, contraction, miss expectations, recessionary language
- **Volatile**: uncertainty, mixed signals, "unclear", forward guidance hedging

**2. `src/data/loader.py`**
- Loads FinancialPhraseBank from HuggingFace datasets
- Applies regime labeling
- Splits into train/val/test (70/15/15), stratified by regime class
- Returns HuggingFace `DatasetDict`
- Exposes `get_class_distribution(split) -> dict` for logging class balance to W&B

**3. `configs/*.yaml`**
Skeleton YAML for each PEFT method. Each has: `method`, `base_model`, `dataset`,
`max_length`, `batch_size`, `learning_rate`, `num_epochs`, `output_dir`.
PEFT-specific fields added per method (e.g., `lora_r`, `lora_alpha` for LoRA).

### Outputs
- `python -c "from src.data.loader import load_dataset; print(load_dataset())"` works
- Class distribution is roughly balanced after labeling

### Tests — `tests/unit/test_data.py`

| Test | What it verifies |
|---|---|
| `test_regime_labels_are_three_classes` | Output dataset has exactly 3 unique labels |
| `test_no_null_text_after_preprocessing` | No null or empty strings in any split |
| `test_stratified_split_preserves_distribution` | Train/val/test class distributions are within 5% of each other |
| `test_get_class_distribution_sums_to_100` | Percentages in distribution dict sum to ~100 |

**Done when:** All 4 tests pass ✅

---

## Phase 2 — Model Loading & PEFT Factory
**Day 2 | ~1.5h**

### Goal
Build a factory that instantiates any of the four PEFT configurations from a config dict,
wrapping either base model.

### What to Build

**1. `src/models/base_loader.py`**
- `load_base_model(model_name, quantize=False)`: loads Llama-3.2-3B or Mistral-7B
- When `quantize=True`: loads in 4-bit via `bitsandbytes` (required for QLoRA)
- Always loads with `device_map="auto"` and attaches the correct tokenizer
- Adds a classification head for 3 classes (Bull/Bear/Volatile)

**2. `src/models/peft_factory.py`**
- `create_peft_model(base_model, config: dict)`: reads `config["method"]` and returns
  the appropriately wrapped PEFT model
- LoRA: `LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], task_type=SEQ_CLS)`
- QLoRA: same LoRA config but base model must have been loaded with `quantize=True`
- Prefix Tuning: `PrefixTuningConfig(num_virtual_tokens=config["num_virtual_tokens"])`
- (IA)³: `IA3Config(task_type=SEQ_CLS)`
- Logs trainable parameter count and % of total params to W&B

### Outputs
- `peft_factory.create_peft_model(base, config)` returns a PEFT model without error
- Trainable param count printed for each method

### Tests — `tests/unit/test_peft_factory.py`

| Test | What it verifies |
|---|---|
| `test_lora_reduces_trainable_params` | LoRA model has fewer trainable params than full fine-tune |
| `test_qlora_requires_quantized_base` | Passing non-quantized base with QLoRA config raises `ValueError` |
| `test_all_methods_produce_valid_model` | All 4 methods instantiate without error (use tiny mock model) |
| `test_trainable_param_count_logged` | `create_peft_model` returns a dict with `trainable_params` key |

**Done when:** All 4 tests pass ✅

---

## Phase 3 — Training Pipeline & W&B Tracking
**Day 3 | ~2h**

### Goal
Build a unified trainer that runs any PEFT method on any base model, with full W&B
experiment tracking.

### What to Build

**1. `src/training/callbacks.py`**
- `WandbCallback`: logs per-epoch train loss, val loss, val accuracy, val F1 (macro),
  trainable param count, GPU memory used, and epoch time

**2. `src/training/trainer.py` — `Trainer` class**
- Reads config YAML
- Calls `load_base_model` + `create_peft_model`
- Uses HuggingFace `Trainer` with the `WandbCallback`
- Saves best checkpoint (by val F1) to `models/{method}_{base_model}_{timestamp}/`
- Saves config as JSON sidecar alongside checkpoint

**3. `scripts/train.py`** — CLI: `--method`, `--base-model`, `--config`, `--seed`

### Outputs
- All 4 PEFT methods train to completion on FinancialPhraseBank
- W&B shows loss + F1 curves per run
- 8 checkpoint directories (4 methods × 2 base models)

### Tests — `tests/unit/test_trainer.py`

| Test | What it verifies |
|---|---|
| `test_trainer_instantiates_from_config` | Config dict → `Trainer` object with correct method set |
| `test_checkpoint_saved_after_short_run` | 10-step mini-run → checkpoint dir exists |
| `test_sidecar_json_contains_method` | JSON sidecar has `method` key matching config |

**`tests/integration/test_short_train.py`** (mark `@pytest.mark.integration`)

| Test | What it verifies |
|---|---|
| `test_lora_trains_50_steps_no_crash` | LoRA + Llama-3.2-3B runs 50 steps without exception |

**Done when:** Unit tests pass; all 8 runs complete with W&B curves ✅

---

## Phase 4 — Custom Evaluation Metrics
**Day 4 | ~1.5h**

### Goal
Implement standard NLP metrics plus the custom **Regime Confidence Score** that
makes this benchmark novel.

### What to Build

**`src/evaluation/metrics.py` — Pure functions**

Standard metrics (wrappers around sklearn/torchmetrics):
- `accuracy(preds, labels) -> float`
- `f1_macro(preds, labels) -> float`
- `per_class_f1(preds, labels) -> dict` — F1 for Bull, Bear, Volatile separately
- `confusion_matrix_dict(preds, labels) -> dict`

Custom metric:
- `regime_confidence_score(preds, labels, probabilities) -> float`
  - Penalizes high-confidence wrong predictions
  - Formula: for each wrong prediction, subtract `max_prob * severity_weight` from
    a base score of 1.0. `severity_weight` for a Bull→Bear error (opposite direction)
    is 2x that of Bull→Volatile (adjacent direction). This reflects real financial risk:
    calling a Bear market Bull is catastrophically worse than calling it Volatile.
  - Returns a score in [0, 1]; higher is better

- `compute_all_metrics(preds, labels, probabilities) -> dict`
  — runs all of the above, returns consistent dict

### Outputs
- `from src.evaluation.metrics import compute_all_metrics` works

### Tests — `tests/unit/test_metrics.py`

| Test | What it verifies |
|---|---|
| `test_perfect_predictions_score_one` | All correct → `regime_confidence_score` = 1.0 |
| `test_opposite_regime_error_penalized_more` | Bull→Bear error reduces score more than Bull→Volatile |
| `test_high_confidence_wrong_penalized_more_than_low` | Same wrong label, higher prob → lower score |
| `test_compute_all_metrics_has_all_keys` | Returns dict with accuracy, f1_macro, per_class_f1, regime_confidence_score |
| `test_per_class_f1_has_three_regimes` | `per_class_f1` dict has keys `bull`, `bear`, `volatile` |

**Done when:** All 5 tests pass ✅

---

## Phase 5 — Benchmark Runner & LoRA Rank Sensitivity
**Day 5 | ~2h**

### Goal
Evaluate all trained models, generate the comparison table, and run the LoRA rank
sensitivity analysis (r = 4, 8, 16, 32, 64).

### What to Build

**1. `src/evaluation/evaluator.py` — `Evaluator`**
Loads each checkpoint, runs inference on the test set, computes `compute_all_metrics`.
Saves to `results/benchmark_results.csv`.

CSV columns:
```
method | base_model | accuracy | f1_macro | bull_f1 | bear_f1 |
volatile_f1 | regime_confidence_score | trainable_params | inference_time_ms
```

**2. `scripts/rank_sensitivity.py`**
Trains LoRA + Llama-3.2-3B five times with r = 4, 8, 16, 32, 64 (all other params fixed).
Logs F1, regime_confidence_score, trainable_params, and training_time_seconds for each r.
Saves results to `results/rank_sensitivity.csv`.
Generates `results/plots/rank_sensitivity.png`: dual-axis line chart with F1 on left y-axis
and trainable param count on right y-axis, r on x-axis. This is your headline research plot.

**3. Plots to generate:**
- `results/plots/f1_comparison.png` — grouped bar chart, F1 by method grouped by base model
- `results/plots/confidence_score_comparison.png` — same structure for regime_confidence_score
- `results/plots/rank_sensitivity.png` — LoRA rank sweep (described above)
- `results/plots/per_class_f1_heatmap.png` — heatmap of per-class F1 across all methods

### Outputs
- `results/benchmark_results.csv` with 8 rows (4 methods × 2 models)
- `results/rank_sensitivity.csv` with 5 rows
- All 4 plots saved
- Real numbers ready for the resume bullet

### Tests — `tests/unit/test_evaluator.py`

| Test | What it verifies |
|---|---|
| `test_csv_has_correct_columns` | Saved CSV contains all expected column names |
| `test_evaluator_handles_missing_checkpoint_gracefully` | Missing checkpoint path → logs warning, skips, does not crash |
| `test_rank_sensitivity_output_has_five_rows` | Rank sweep CSV has exactly 5 rows |

**Done when:** All tests pass; all plots generated; real numbers in hand ✅

---

## Phase 6 — Gradio App & Hugging Face Spaces Deployment
**Day 6 | ~1.5h**

### Goal
Build a Gradio demo where users paste financial text and see regime predictions from the
best model, then deploy to Hugging Face Spaces.

### What to Build

**`frontend/app.py` — Gradio App (two tabs)**

**Tab 1: Live Regime Classifier**
- Textbox for financial text input (placeholder: example Fed statement)
- Dropdown to select model (best LoRA, best QLoRA, etc.)
- "Classify" button → runs inference, returns:
  - Predicted regime with colored badge (🟢 Bull / 🔴 Bear / 🟡 Volatile)
  - Confidence bar for each class
  - Regime Confidence Score for this prediction
  - A one-line interpretation: "This text exhibits Bear characteristics with high confidence"

**Tab 2: Benchmark Results**
- Static display of `results/benchmark_results.csv` as a `gr.Dataframe`
- The rank sensitivity plot displayed with `gr.Image`
- One-line callout of the best method and its regime_confidence_score

**`README_HF.md`** — HF Spaces metadata YAML header + one-paragraph project description.

### Outputs
- `python frontend/app.py` launches locally
- HF Space live at `https://huggingface.co/spaces/{username}/peft-regime-benchmark`

### Tests — `tests/unit/test_frontend.py`

| Test | What it verifies |
|---|---|
| `test_regime_badge_bull` | `regime_badge("bull")` returns string containing `"🟢"` |
| `test_regime_badge_bear` | `regime_badge("bear")` returns string containing `"🔴"` |
| `test_interpretation_text_not_empty` | `generate_interpretation("bear", 0.91)` returns non-empty string |

**Done when:** All 3 tests pass; HF Space live ✅

---

## Phase 7 — CI/CD & Documentation
**Day 7 | ~1.5h**

### Goal
GitHub Actions, final test suite, and recruiter-ready README with real numbers.

### What to Build

**1. `tests/unit/test_end_to_end_structure.py`**

| Test | What it verifies |
|---|---|
| `test_all_configs_loadable` | All 4 YAML configs load without error |
| `test_metrics_and_evaluator_compatible` | `compute_all_metrics` output format matches evaluator's expected input |

**2. `pytest.ini`** — marks for `unit`/`integration`, default `tests/unit/`, 70% coverage floor

**3. `.github/workflows/ci.yml`** — two jobs:
- `lint`: `ruff check` + `black --check` on every push and PR
- `unit-tests`: Python 3.10, install requirements (with dummy env vars), `pytest tests/unit/`

**4. `README.md`** — sections:
- Header + badges + HF Space live demo link
- The financial problem (why regime detection matters, consequences of misclassification)
- Architecture: PEFT methods diagram, custom Regime Confidence Score explanation
- **Benchmark results table** (fill with real numbers)
- **LoRA rank sensitivity plot** (embed `results/plots/rank_sensitivity.png`)
- Key findings: 3 bullets from your actual results
- Reproduction: `python scripts/train.py --method lora --base-model llama3 --seed 42`
  runs under $5 compute cost using QLoRA

**Final checklist:**
- [ ] CI green on main
- [ ] README benchmark table filled with real numbers
- [ ] W&B project public and linked
- [ ] HF Space demo accessible

**Done when:** CI green; README live; HF demo accessible ✅

---

## Resume Bullet

> Conducted a **systematic PEFT benchmark** comparing LoRA, QLoRA, Prefix Tuning, and
> (IA)³ on **market regime detection** (Bull/Bear/Volatile) across Llama-3.2-3B and
> Mistral-7B, introducing a custom **Regime Confidence Score** metric that penalizes
> high-confidence misclassifications; performed LoRA rank sensitivity analysis (r=4–64);
> all experiments reproducible under **$5 compute cost** via W&B tracking, deployed to
> **Hugging Face Spaces**.

---

## Daily Budget

| Day | Phase | Hours |
|---|---|---|
| 1 | Data Pipeline | 1.5h |
| 2 | Model + PEFT Factory | 1.5h |
| 3 | Training Pipeline | 2h |
| 4 | Custom Metrics | 1.5h |
| 5 | Benchmark + Rank Sweep | 2h |
| 6 | Gradio + Deploy | 1.5h |
| 7 | CI + README | 1.5h |

**Total: ~11.5 hours**
