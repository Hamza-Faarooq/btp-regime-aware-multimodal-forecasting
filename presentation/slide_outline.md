# Presentation outline (18-slide defence deck; target 10-15 minutes)

1. Cover: title (use the registered title), supervisor, one-sentence question.
2. Answer first: summary of findings.
3. Gap: why evaluation protocol and news value matter.
4. Data: NIFTY / SPY, split timeline, label = next-day move, class and headline-volume shift.
5. Data audit: label timing check, recovered daily series (2,695 days; 1 genuinely missing day, 2018-12-31), LAG = 0 convention.
6. Pipeline: five scripts; 240 evaluations (180 trained + 60 baseline).
7. Models M0-M6 (M7 implemented, not run).
8. Regimes: label-mix differences as associations (High-vol = high volatility after falling prices, not a bearish signal).
9. Three protocols; P1 test sample differs from P2/P3.
10. Main result (P3 Macro-F1 vs the random-guess floor).
11. Diagnostics: overfitting, predicted class shares.
12. RQ1: protocol gaps (seed bootstrap, 2,000 resamples; M5 the only positive case; mechanisms untested).
13. RQ2: news value by regime and the 2020 stress window (row bootstrap, 1,000 resamples; 4 of 24 intervals exclude zero, exploratory).
14. Mechanisms (hypotheses, untested): overfitting, low day-to-day embedding variation, era shift.
15. Validity and limitations (timing, multiplicity, post-hoc M0r/M6, row bootstrap ignores temporal dependence).
16. Conclusion.
17. References.
18. Thank you.

Protocol caveat to state aloud: P1, P2 and P3 do not share the same test sample, so score gaps combine protocol changes with sample-composition differences; P2 vs P3 is the cleaner contrast.

Likely viva questions: why Macro-F1? why is the majority baseline low on Macro-F1? why did embeddings overfit? what exactly is leakage here? why is headline publication timing uncertain? what was added after seeing results (M0r, M6)? why are M7 and LAG=1 not reported as results? why are the row-bootstrap intervals probably too narrow?
