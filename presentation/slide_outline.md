# Presentation outline (16-slide defence deck; target 10–15 minutes)

1. Title, supervisor, one-sentence question.
2. Motivation: reported market-prediction scores depend on how they are evaluated, and news may help only in some regimes.
3. Research questions RQ1 (evaluation sensitivity), RQ2 (news value by regime).
4. Data: NIFTY / SPY, split timeline (`fig_dataset_timeline.png`), label = next-day move.
5. Data audit: recovered daily series, label timing check, class shift, news volume shift.
6. Models M0-M6 (`fig_architecture.png`).
7. Regimes (`regimes_overview.png`): present label-mix differences as associations, not proof that the model mainly predicts volatility magnitude.
8. Three protocols (`fig_protocol_design.png`).
9. Result 1: main table (P3 test Macro-F1) and the random-guess floor.
10. Result 2: protocol gaps (`fig_protocol_audit.png`): small and model-dependent.
11. Result 3: news by regime and the 2020 stress window (`fig_regime_news_value.png`); embedding hurts, sentiment about neutral.
12. Limitations (small test set, many comparisons, post-hoc M6, news volume confound) and conclusion.

Protocol caveat to state aloud: P1, P2 and P3 do not share the same test sample, so score gaps combine protocol changes with sample-composition differences; they are not a clean causal estimate of leakage.

Likely viva questions: why Macro-F1? why is the majority baseline low on Macro-F1? why did embeddings overfit? what exactly is leakage here? why is headline publication timing uncertain? what was added after seeing results (M6)? why are M7 and LAG=1 not reported as results?
