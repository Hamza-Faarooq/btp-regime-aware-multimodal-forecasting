# Evaluation Robustness in Multimodal Financial Market Prediction

## Undergraduate Thesis — IIT Kharagpur

This repository contains the implementation and experimental record for a bachelor's thesis investigating the reliability of multimodal financial market prediction under different evaluation protocols and market conditions.

The central premise of the study is that reported predictive performance can depend not only on the model architecture, but also on **how the experiment is constructed and evaluated**. The project therefore treats evaluation protocol as a first-class experimental variable and examines whether financial news contributes information beyond historical market data, particularly during periods of elevated market stress.

---

## Research Questions

### RQ1 — Evaluation sensitivity

**How much do reported market-prediction results change when the same models are evaluated under increasingly strict temporal protocols?**

The study compares model performance under three protocols:

- **P1 — Random / permissive:** random splitting with preprocessing performed using the complete dataset. This serves as an intentionally permissive reference and is not treated as a valid deployment-style evaluation.
- **P2 — Chronological:** temporally ordered train/test evaluation while retaining broader preprocessing.
- **P3 — Strict chronological:** temporally ordered evaluation with preprocessing restricted to the training data, providing the primary leakage-controlled benchmark.

The objective is not simply to identify the highest-performing protocol, but to quantify how much the apparent performance of the same models changes as evaluation becomes more rigorous.

### RQ2 — Incremental value of financial news

**Does financial news provide predictive information beyond historical price data, and does its contribution vary across market regimes and periods of market stress?**

News is represented using frozen FinBERT-derived features and evaluated alongside price-based and regime-aware models. Performance is examined both over the full test period and within distinct volatility regimes, with a dedicated analysis of the February–April 2020 stress window.

---

## Experimental Design

The study uses a controlled model family so that differences in results can be attributed to information sources and evaluation choices rather than an uncontrolled collection of architectures.

| Model | Information used | Purpose |
|---|---|---|
| **M0** | Majority class | Deterministic baseline |
| **M0r** | Random prediction | Chance-level reference |
| **M1** | Historical price | Price-only benchmark |
| **M2** | Price + FinBERT news representation | Multimodal benchmark |
| **M3** | Price + news + market regime | Regime-aware multimodal model |
| **M4** | News representation | News-only ablation |
| **M5** | Market regime | Regime-only ablation |
| **M6** | Price + FinBERT sentiment features | Compact news-fusion model |

The final experiments use **10 fixed random seeds (0–9)** to assess variability across runs. Model settings are fixed rather than tuned against the test set.

---

## Market Regimes

Market conditions are represented using a three-state volatility regime formulation derived from rolling market-return and volatility characteristics:

- **Low-volatility**
- **Mid-volatility**
- **High-volatility**

This allows the study to distinguish overall predictive performance from performance conditional on the state of the market.

The regime analysis is particularly important for RQ2: a news signal that provides little incremental information during ordinary periods may behave differently when markets experience unusually large movements and uncertainty.

---

## Evaluation Framework

A central component of the thesis is the explicit separation between **model performance** and **evaluation validity**.

The same model family is evaluated under multiple protocols, allowing the resulting performance differences to be interpreted as an evaluation effect rather than a model effect.

The strict protocol is the primary benchmark for scientific conclusions. Results from more permissive protocols are retained as part of the audit rather than being presented as equally valid estimates of out-of-sample performance.

Performance is evaluated using **macro-F1**, which is appropriate for comparing predictive quality across market-movement classes without allowing the majority class to dominate the headline metric.

---

## Current Results

The executed strict-protocol benchmark provides the following test-set macro-F1 results, reported as mean ± standard deviation across 10 seeds:

| Model | Macro-F1 |
|---|---:|
| M0 — Majority | 0.207 ± 0.000 |
| M0r — Random | 0.333 ± 0.020 |
| M1 — Price | 0.466 ± 0.036 |
| M2 — Price + News | 0.416 ± 0.023 |
| M3 — Price + News + Regime | 0.433 ± 0.025 |
| M4 — News only | 0.382 ± 0.037 |
| M5 — Regime only | 0.354 ± 0.019 |
| M6 — Price + Sentiment | 0.482 ± 0.038 |

These results do **not** support a simple claim that adding a high-dimensional news embedding automatically improves prediction. In the current experiment, the price-only model provides a stronger benchmark than the full price-plus-news embedding model, while the compact sentiment representation achieves the highest observed strict-protocol macro-F1 among the tested predictive models.

The protocol audit likewise shows that the effect of evaluation choices is **model-dependent rather than uniformly inflationary**. This distinction is central to the thesis: the study is designed to measure evaluation sensitivity rather than assume that every permissive protocol necessarily produces a larger score.

---

## Stress-Period Analysis

A dedicated stress-period analysis focuses on **February–April 2020**, covering the major market disruption surrounding the COVID-19 crash.

The analysis compares model performance during the stress window with performance outside the window and further examines performance across volatility regimes.

This is intended to answer the more specific question:

> Does the usefulness of financial news change when the market itself enters an unusual regime?

The stress analysis is therefore treated as a conditional analysis of the news signal, rather than as evidence that a model is universally superior.

---

## Repository Structure

```text
btp-regime-aware-multimodal-forecasting/
│
├── data/                  # Input and processed datasets
├── figures/               # Generated research figures
├── literature/            # Literature notes and references
├── notebooks/             # Executable Colab/Jupyter notebooks
├── report/                # Thesis report materials
├── presentation/          # Presentation materials
├── results/               # Metrics, predictions and result summaries
│
├── src/
│   ├── environment.py
│   ├── data.py
│   ├── regimes.py
│   ├── finbert.py
│   ├── model_training.py
│   ├── audit.py
│   ├── analysis.py
│   └── figures.py
│
├── config.py              # Frozen experiment configuration
└── references.md          # Literature review and research-gap notes
```

The `notebooks/` directory contains the execution-oriented notebooks used during the research workflow. The corresponding reusable Python pipeline code is maintained separately under `src/`.

---

## Reproducibility

The project follows a fixed experimental configuration:

- **10 random seeds:** 0–9
- Fixed model and training configuration
- Chronological evaluation for the strict benchmark
- Training-only preprocessing under the strict protocol
- No test-set hyperparameter tuning
- Frozen FinBERT representations rather than repeatedly fine-tuning the language model

The purpose of these constraints is to make the comparison between protocols and information sources interpretable and reproducible.

---

## Research Status

The main data-processing, regime construction, FinBERT representation, model training, protocol audit, regime analysis, and stress-period analysis pipelines have been implemented and executed.

Two robustness extensions remain explicitly planned before the experimental phase is considered closed:

1. **Regime-conditioned / gated fusion** to test whether the usefulness of news can be made conditional on market state.
2. **Lagged news analysis (LAG=1)** to provide a stricter temporal robustness check on the news signal.

These experiments are intended as robustness checks rather than changes to the research questions or core experimental design.

---

## Scope and Interpretation

This project is an undergraduate research study, not a trading system or investment strategy.

The objective is to study **evaluation reliability and conditional predictive information** in multimodal market prediction. Reported predictive performance should therefore be interpreted as an experimental result under the stated data construction and evaluation protocols, not as evidence of exploitable financial returns or deployable trading performance.

The final thesis will distinguish clearly between:

- observed empirical results,
- methodological implications,
- robustness evidence,
- and limitations of the experimental setup.

---

## Citation and Academic Use

This repository accompanies an undergraduate thesis at **IIT Kharagpur**. The final report and presentation will provide the complete methodological description, literature context, experimental results, limitations, and conclusions.

For literature sources and the research-gap formulation, see [references.md](references.md).
