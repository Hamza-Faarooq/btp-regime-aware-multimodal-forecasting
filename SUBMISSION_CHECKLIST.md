# BTP submission checklist — target: 25 October 2026

This checklist is intentionally limited to finishing and validating the already chosen project. It does not change the research questions or add unexecuted experiments to the results.

## Required before submission

- [ ] Compile `report/latex/main.tex` from the `report/latex/` directory twice with pdfLaTeX; resolve missing figures, undefined references, overfull pages and table-of-contents issues.
- [ ] Replace `report/BTP_Report.pdf` with the newly compiled PDF only after visually checking every page.
- [ ] Verify title, student details, degree/program, supervisor name, and institute-required certificate/declaration format. Do not add a supervisor signature without approval.
- [ ] Check all result values against `results/EXECUTED_RESULTS.md`; do not present M7 or LAG=1 as executed.
- [ ] Keep the P1/P2/P3 caveat prominent: the test samples differ, so protocol gaps are not a pure causal estimate of leakage.
- [ ] Keep the headline-timing limitation prominent: headline timestamps are dates only; availability by the close cannot be verified.
- [ ] Make the literature chapter accurately describe the scope of the papers reviewed. Verify bibliographic details and remove or qualify any citation whose details cannot be confirmed.
- [ ] Read the final PDF from start to finish and check that each table/figure is legible, numbered, referenced in the text, and consistent with the discussion.
- [ ] Finalise the defence deck against the compiled PDF; check all numbers, terminology, slide readability, and time.
- [ ] Rehearse a 10–15 minute presentation and practise explaining the target, split design, baselines, confidence intervals, post-hoc M6, and limitations.

## Research claims to preserve

- The P3 price-only LSTM obtains Macro-F1 0.466.
- Price + FinBERT embedding obtains 0.416; difference from price-only is -0.050 (95% bootstrap interval [-0.087, -0.012]).
- Price + sentiment obtains 0.482, but the difference from price-only (+0.016; interval [-0.017, +0.049]) is uncertain.
- The stress subset has 50 rows; no reliable news advantage is detected, which is not proof of no effect.
- Looser protocols do not inflate all models. Regime-only M5 is the clearest positive protocol-gap case.
- The report is an evaluation audit on one benchmark, not a trading system or a claim of profitability.

## Important reproducibility note

The recorded notebook outputs and summary tables have been inspected for headline-level consistency, but a clean end-to-end reproduction from raw data has not yet been independently verified. Do not claim otherwise.
