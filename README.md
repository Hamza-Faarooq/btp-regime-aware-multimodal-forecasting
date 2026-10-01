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
- M4: News-only secondary model

## Evaluation protocols
- **P1:** Random shuffle with all-data preprocessing — deliberately invalid temporal protocol
- **P2:** Chronological split with all-data preprocessing
- **P3:** Chronological split with train-only preprocessing — strict protocol

## Reproducibility
Fixed seeds: `0, 1, 2`.

Hyperparameters are stored in `config.py` and are fixed in advance; no test-set tuning.

## Day 0 status
- Repository skeleton: complete
- Fixed configuration: complete
- Literature review: provisionally complete
- Provisional research gap: recorded in `references.md`
- Final Step 0 closure: complete

See **[references.md](references.md)** for the literature notes, source links, synthesis, and finalized Day-0 gap.

## Workflow
The repository follows the finalized BTP Project Brief step by step. Do not move to a later step until the current checkpoint passes.
