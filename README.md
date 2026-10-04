# Regime-Aware Multimodal Deep Learning for Financial Market Forecasting

BTP project: evaluation robustness and regime-dependent news value.

## Research questions
- **RQ1:** How sensitive are multimodal market-movement prediction results to the evaluation protocol?
- **RQ2:** Does financial news provide incremental predictive information beyond market history, and does that contribution differ across market regimes?

## Models
- M0: Majority-class baseline
- M1: Price LSTM
- M2: Price + frozen FinBERT
- M3: Price + News + Regime
- M4: News-only model
- M5: Regime-only model
- M6: Price + 3-number FinBERT sentiment

## Evaluation protocols
- **P1:** Random shuffle with all-data preprocessing — deliberately invalid temporal protocol
- **P2:** Chronological split with all-data preprocessing
- **P3:** Chronological split with train-only preprocessing — strict protocol

## Reproducibility
Fixed seeds for the final run: `0–9` (10 seeds).

Hyperparameters are stored in `config.py` and are fixed in advance; no test-set tuning.

## Current executed status
- Repository skeleton: complete
- Fixed configuration: complete
- Literature review: complete
- Provisional research gap: recorded in `references.md`
- Main experiment: executed through analysis

See **[references.md](references.md)** for the literature notes, source links, synthesis, and finalized Day-0 gap.

## Workflow
The repository now records the executed working notebook state. Two explicitly planned robustness additions (regime-conditioned gating and LAG=1) remain pending execution.
