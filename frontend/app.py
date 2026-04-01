"""
Gradio app for the PEFT Regime Benchmark.

Tab 1: Live Regime Classifier — paste financial text, select a model, get regime prediction.
Tab 2: Benchmark Results — static comparison table and rank sensitivity plot.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES = {0: "Bull", 1: "Bear", 2: "Volatile"}
LABEL_BADGES = {"bull": "🟢 Bull", "bear": "🔴 Bear", "volatile": "🟡 Volatile"}

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = Path(__file__).parent.parent / "models"

EXAMPLE_TEXTS = [
    "The Federal Reserve raised interest rates by 25 basis points, signaling continued confidence in economic growth and robust labor market conditions.",
    "Company reported a significant decline in quarterly earnings, missing analyst estimates by 30%, as consumer demand weakened sharply.",
    "Mixed economic signals persist as inflation remains elevated while unemployment data showed unexpected improvement, leaving markets uncertain about the Fed's next move.",
]

# ---------------------------------------------------------------------------
# Utility functions (module-level for testability)
# ---------------------------------------------------------------------------


def regime_badge(regime: str) -> str:
    """Return an emoji badge string for a regime label."""
    return LABEL_BADGES.get(regime.lower(), f"⬜ {regime.capitalize()}")


def generate_interpretation(regime: str, confidence: float) -> str:
    """Return a one-line interpretation string for a regime prediction."""
    level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
    descriptions = {
        "bull": "bullish/positive market",
        "bear": "bearish/negative market",
        "volatile": "uncertain/mixed market",
    }
    desc = descriptions.get(regime.lower(), regime.lower() + " market")
    return f"This text exhibits {desc} characteristics with {level} confidence ({confidence:.1%})."


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_checkpoint(method_dir: Path) -> Path | None:
    """Find the single checkpoint subdirectory inside a method folder."""
    if not method_dir.exists():
        return None
    subdirs = [d for d in method_dir.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()]
    if not subdirs:
        return None
    # Pick the most recently modified
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def _discover_models() -> dict[str, Path]:
    """
    Scan models/ and return {display_name: checkpoint_path} for available checkpoints.
    Ordered by benchmark F1 (best first).
    """
    candidates = [
        ("QLoRA — 90.8% F1 (best accuracy)", MODELS_DIR / "qlora_llama3"),
        ("LoRA — 88.9% F1", MODELS_DIR / "lora_llama3"),
        ("(IA)³ — 87.6% F1, most parameter-efficient", MODELS_DIR / "ia3_llama3"),
    ]
    available = {}
    for label, method_dir in candidates:
        ckpt = _find_checkpoint(method_dir)
        if ckpt is not None:
            available[label] = ckpt
    return available


AVAILABLE_MODELS = _discover_models()

# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Loads and caches a PEFT model for single-sample inference.

    Reloads only when the selected checkpoint changes.
    """

    def __init__(self):
        self._current_ckpt: Path | None = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._max_length = 128

    def _load(self, ckpt_path: Path):
        """Load base model + PEFT adapter from checkpoint."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

        from src.data.preprocessor import ID2LABEL, LABEL2ID

        sidecar = ckpt_path / "training_config.json"
        with open(sidecar) as f:
            cfg = json.load(f)

        model_name = cfg.get("model_name_or_path", "meta-llama/Llama-3.2-3B")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={0: "3500MiB", "cpu": "12GiB"},
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
        else:
            base = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )

        base.config.pad_token_id = base.config.eos_token_id
        model = PeftModel.from_pretrained(base, str(ckpt_path))

        if device == "cuda":
            for param in model.parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.float16)

        model.eval()
        self._model = model
        self._current_ckpt = ckpt_path

    def predict(self, text: str, ckpt_path: Path) -> dict:
        """
        Run inference on a single text string.

        Returns dict with keys: regime, confidence, probabilities, rcs, interpretation, latency_ms
        """
        import torch

        if ckpt_path != self._current_ckpt:
            self._load(ckpt_path)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._device == "cuda"
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        )

        t0 = time.time()
        with torch.no_grad(), autocast_ctx:
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        latency_ms = (time.time() - t0) * 1000

        logits = outputs.logits.float().cpu().numpy()[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        pred_idx = int(np.argmax(probs))
        pred_label = LABEL_NAMES[pred_idx].lower()
        confidence = float(probs[pred_idx])

        # Single-sample RCS: perfect if correct (unknown), so just show confidence-based score
        # We report 1 - (max_prob_wrong * severity) as a per-sample risk signal
        # Since we don't have ground truth, report the confidence as the RCS proxy
        rcs_display = f"{confidence:.3f} (confidence proxy — no ground truth for single sample)"

        return {
            "regime": pred_label,
            "confidence": confidence,
            "probabilities": {
                "Bull": float(probs[0]),
                "Bear": float(probs[1]),
                "Volatile": float(probs[2]),
            },
            "rcs": rcs_display,
            "interpretation": generate_interpretation(pred_label, confidence),
            "latency_ms": latency_ms,
        }


_engine = InferenceEngine()

# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def classify(text: str, model_name: str):
    """Gradio callback: run inference and format outputs."""
    if not text.strip():
        return "Please enter some text.", "", "", "", ""

    if not AVAILABLE_MODELS:
        return "No trained models found in models/. Train first.", "", "", "", ""

    if model_name not in AVAILABLE_MODELS:
        return f"Model '{model_name}' not found.", "", "", "", ""

    ckpt_path = AVAILABLE_MODELS[model_name]

    try:
        result = _engine.predict(text, ckpt_path)
    except Exception as e:
        return f"Inference error: {e}", "", "", "", ""

    badge = regime_badge(result["regime"])
    probs = result["probabilities"]
    confidence_str = (
        f"Bull:     {probs['Bull']:.1%}\n"
        f"Bear:     {probs['Bear']:.1%}\n"
        f"Volatile: {probs['Volatile']:.1%}"
    )
    rcs = result["rcs"]
    interpretation = result["interpretation"]
    latency = f"{result['latency_ms']:.1f} ms"

    return badge, confidence_str, rcs, interpretation, latency


# ---------------------------------------------------------------------------
# Benchmark data helpers
# ---------------------------------------------------------------------------

def _load_benchmark_df() -> pd.DataFrame:
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        return pd.DataFrame({"message": ["results/benchmark_results.csv not found"]})
    df = pd.read_csv(csv_path)
    # Round floats for display
    float_cols = ["accuracy", "f1_macro", "bull_f1", "bear_f1", "volatile_f1", "regime_confidence_score"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    return df


def _best_method_callout(df: pd.DataFrame) -> str:
    if "f1_macro" not in df.columns or df.empty:
        return ""
    try:
        numeric = df.copy()
        numeric["f1_macro"] = pd.to_numeric(numeric["f1_macro"], errors="coerce")
        best = numeric.loc[numeric["f1_macro"].idxmax()]
        method = best["method"].upper()
        f1 = float(best["f1_macro"])
        rcs = best.get("regime_confidence_score", "N/A")
        return f"Best method: **{method}** — F1 Macro: {f1:.4f} | Regime Confidence Score: {rcs}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr

    model_choices = list(AVAILABLE_MODELS.keys()) if AVAILABLE_MODELS else ["No models found"]
    benchmark_df = _load_benchmark_df()
    best_callout = _best_method_callout(benchmark_df)

    rank_plot_path = str(RESULTS_DIR / "plots" / "rank_sensitivity.png")
    rank_plot_exists = Path(rank_plot_path).exists()

    with gr.Blocks(title="PEFT Regime Benchmark", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# PEFT Regime Benchmark — Market Regime Detector
Comparing **LoRA**, **QLoRA**, **Prefix Tuning**, and **(IA)³** on financial text classification.
Regimes: 🟢 Bull · 🔴 Bear · 🟡 Volatile
"""
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1: Live classifier
            # ------------------------------------------------------------------
            with gr.Tab("Live Regime Classifier"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Financial Text",
                            placeholder="Paste a Fed statement, earnings call excerpt, or macro news headline…",
                            lines=5,
                        )
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0],
                            label="Model",
                        )
                        classify_btn = gr.Button("Classify", variant="primary")

                        gr.Examples(
                            examples=EXAMPLE_TEXTS,
                            inputs=text_input,
                            label="Example texts",
                        )

                    with gr.Column(scale=1):
                        badge_out = gr.Textbox(label="Predicted Regime", interactive=False)
                        confidence_out = gr.Textbox(
                            label="Class Probabilities", lines=3, interactive=False
                        )
                        rcs_out = gr.Textbox(label="Regime Confidence Score", interactive=False)
                        interp_out = gr.Textbox(label="Interpretation", interactive=False)
                        latency_out = gr.Textbox(label="Inference Latency", interactive=False)

                classify_btn.click(
                    fn=classify,
                    inputs=[text_input, model_dropdown],
                    outputs=[badge_out, confidence_out, rcs_out, interp_out, latency_out],
                )

            # ------------------------------------------------------------------
            # Tab 2: Benchmark results
            # ------------------------------------------------------------------
            with gr.Tab("Benchmark Results"):
                if best_callout:
                    gr.Markdown(best_callout)

                gr.Markdown("### Method Comparison (Llama-3.2-3B, test set)")
                gr.Dataframe(value=benchmark_df, interactive=False)

                gr.Markdown(
                    """
**Notes:**
- Prefix Tuning result reflects model failure (incompatible with 4-bit quantization + Llama-3.2 DynamicCache)
- (IA)³ achieves competitive F1 with ~15× fewer trainable params than LoRA/QLoRA
- Regime Confidence Score (RCS) penalizes high-confidence wrong predictions — Bull↔Bear errors weighted 2×
"""
                )

                if rank_plot_exists:
                    gr.Markdown("### LoRA Rank Sensitivity (r = 4, 8, 16, 32, 64 | 30 steps each)")
                    gr.Image(
                        value=rank_plot_path,
                        label="F1 vs Trainable Parameters across LoRA ranks",
                        interactive=False,
                    )
                    gr.Markdown(
                        "**Finding:** F1 improves steeply from r=4→32, then plateaus — r=64 yields no gain "
                        "despite 2× the parameters. r=32 is the crossover point for this task."
                    )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
