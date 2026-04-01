---
title: PEFT Regime Benchmark
emoji: 📈
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: frontend/app.py
pinned: false
license: mit
---

# PEFT Regime Benchmark — Market Regime Detector

A rigorous benchmark comparing **LoRA**, **QLoRA**, **Prefix Tuning**, and **(IA)³** PEFT fine-tuning methods on financial market regime detection using Llama-3.2-3B.

## What it does

Classifies financial text (Fed statements, earnings calls, macro news) into three market regimes:
- 🟢 **Bull** — positive outlook, growth language, beat expectations
- 🔴 **Bear** — negative outlook, contraction, miss expectations
- 🟡 **Volatile** — uncertainty, mixed signals, forward guidance hedging

## Novel contribution

A custom **Regime Confidence Score** that penalizes high-confidence wrong predictions — Bull↔Bear errors weighted 2× more than adjacent errors, reflecting real financial risk where calling a Bear market Bull is catastrophically worse than calling it Volatile.

## Results (Llama-3.2-3B, FinancialPhraseBank)

| Method | Accuracy | F1 Macro | RCS | Trainable Params |
|--------|----------|----------|-----|-----------------|
| **QLoRA** | 90.0% | 90.8% | 0.957 | 4,596,736 |
| LoRA | 87.8% | 88.9% | 0.951 | 4,596,736 |
| (IA)³ | 87.0% | 87.6% | 0.942 | 295,936 |
| Prefix Tuning | 44.1% | 33.7% | — | 582,656 |

**(IA)³ achieves competitive accuracy with ~15× fewer parameters than LoRA/QLoRA.**

## Reproduce locally

```bash
git clone https://github.com/<your-username>/peft-regime-benchmark
cd peft-regime-benchmark
pip install -r requirements.txt
python scripts/train.py --method qlora --base-model llama3 --seed 42
python scripts/evaluate.py --checkpoints models/qlora_llama3/*
python frontend/app.py
```
