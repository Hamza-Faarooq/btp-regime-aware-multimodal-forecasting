# Draft text: Abstract, Results, Discussion, Limitations, Conclusion

> Every number below comes from your final 10-seed run (Cell 1 and Cell 3 output of `step6_analysis.py`).
> Items in [square brackets] are for you to fill in or check. Rewrite in your own words and make sure you can explain every sentence.

---

## Abstract (draft)

This thesis studies next-day market-movement classification (Rise / Neutral / Fall) on the NIFTY benchmark (SPY, 2010-2020) using lagged market data, financial news headlines encoded with a frozen FinBERT model, and data-driven market regimes obtained by clustering rolling return and volatility. It asks two questions: how sensitive reported performance is to the evaluation protocol, and whether news and regime information add predictive value beyond lagged prices, in particular across regimes and during the Feb-Apr 2020 stress period. Under a strict chronological protocol with train-only preprocessing, all models beat a random-guess floor (Macro-F1 0.333) but a price-only LSTM (0.466) was not improved by 768-dimensional FinBERT embeddings (0.416, a significant decrease), by a three-number FinBERT sentiment input (0.482, +0.016, interval including zero) or by an explicit regime input. Relaxing the protocol (random splits, preprocessing fitted on all data) produced no systematic optimism, with one exception: a regime-only model gained 0.046 under a random split. We conclude that on this benchmark the incremental value of news is not reliably demonstrated: the embedding model performs worse than price-only, while the sentiment model's small point-estimate gain is uncertain. The regime-conditioned class distributions suggest an association between recent volatility and the next-day label distribution, but these experiments do not establish that most predictive skill comes from identifying next-day volatility rather than direction.

---

## 5. Results

### 5.1 Reporting conventions
All models were trained with 10 random seeds. Test Macro-F1 is averaged over seeds. Confidence intervals for differences between models were obtained by resampling test rows 1000 times (the same resample for all models, seed-averaged score), so they reflect test-set sampling uncertainty. The test set has 317 rows; the strict-protocol regimes have 125 (Low-vol), 113 (Mid-vol) and 79 (High-vol) test rows, and the stress window (1 Feb - 30 Apr 2020) contains 50 rows. M0 always predicts the majority class; M0r guesses uniformly at random and is the appropriate floor for Macro-F1 (a majority predictor scores low on Macro-F1 by construction).

### 5.2 Overall comparison under the strict protocol (P3)

**Table 1. Test Macro-F1, strict protocol (mean +/- sd over 10 seeds).**

| Model | Inputs | Macro-F1 | Accuracy | Balanced acc. |
|---|---|---|---|---|
| M0 | majority class | 0.207 | 0.451 | 0.333 |
| M0r | uniform random guess | 0.333 +/- 0.020 | 0.339 | 0.341 |
| M1 | price window (LSTM) | 0.466 +/- 0.036 | 0.530 | 0.476 |
| M2 | price + FinBERT embedding | 0.416 +/- 0.023 | 0.493 | 0.448 |
| M3 | price + FinBERT embedding + regime | 0.433 +/- 0.025 | 0.491 | 0.454 |
| M4 | FinBERT embedding only | 0.382 +/- 0.037 | 0.402 | 0.394 |
| M5 | regime only | 0.354 +/- 0.019 | 0.493 | 0.415 |
| M6 | price + FinBERT sentiment (3 numbers) | 0.482 +/- 0.038 | 0.531 | 0.498 |

All learned models exceed the random floor. The price-only model is competitive with every richer model, and the model using only the regime label performs close to the floor, so the regime label alone carries little usable information.

### 5.3 Effect of the evaluation protocol (RQ1)

**Table 2. Test Macro-F1 by protocol (mean over 10 seeds).**

| Model | P1 random split, fit on all | P2 chronological, fit on all | P3 strict |
|---|---|---|---|
| M1 price | 0.438 | 0.460 | 0.466 |
| M2 price+news (emb) | 0.423 | 0.428 | 0.416 |
| M3 price+news+regime | 0.421 | 0.441 | 0.433 |
| M4 news only | 0.363 | 0.369 | 0.382 |
| M5 regime only | 0.400 | 0.382 | 0.354 |
| M6 price+sentiment | 0.431 | 0.448 | 0.482 |

**Table 3. Optimistic performance gap (looser protocol minus P3), with 95% bootstrap interval over seeds.**

| Model | P1 - P3 | P2 - P3 |
|---|---|---|
| M1 | -0.028 [-0.055, 0.001] | -0.005 [-0.034, 0.023] |
| M2 | +0.006 [-0.010, 0.022] | +0.012 [-0.008, 0.031] |
| M3 | -0.012 [-0.030, 0.008] | +0.008 [-0.010, 0.028] |
| M4 | -0.019 [-0.040, 0.004] | -0.013 [-0.043, 0.018] |
| M5 | **+0.046 [0.029, 0.063]** | **+0.028 [0.007, 0.048]** |
| M6 | -0.051 [-0.079, -0.019] | -0.034 [-0.071, 0.005] |

For models that use lagged prices and/or news, the loosely specified protocols did not score higher than the strict one; most intervals include zero. The exception is the regime-only model (M5), which scored higher under both looser protocols. Regimes persist for weeks, so a random split places near neighbours of every test day in the training set, and fitting the clusters on all data lets test-period structure shape the labels. For M6 the strict protocol scored *higher* than the random split; this cannot be a leakage effect and likely reflects differences between the test sets of the three protocols and run-to-run variability.

### 5.4 Value of news and regime information (RQ2, strict protocol)

**Table 4. Test Macro-F1 by group (strict protocol, seed-averaged).**

| Group | n | Random floor | M1 price | M2 +emb | M3 +emb+regime | M6 +sentiment |
|---|---|---|---|---|---|---|
| All test | 317 | 0.333 | 0.466 | 0.416 | 0.433 | 0.482 |
| Low-vol | 125 | 0.329 | 0.409 | 0.386 | 0.407 | 0.415 |
| Mid-vol | 113 | 0.322 | 0.410 | 0.341 | 0.361 | 0.464 |
| High-vol | 79 | 0.323 | 0.357 | 0.363 | 0.319 | 0.378 |
| Stress window (Feb-Apr 2020) | 50 | 0.316 | 0.473 | 0.448 | 0.450 | 0.471 |
| Outside stress window | 267 | 0.334 | 0.451 | 0.391 | 0.416 | 0.467 |

**Table 5. Differences in Macro-F1 (second minus first model), 95% bootstrap interval from resampling test rows.**

| Group | M2 - M1 (embedding news) | M6 - M1 (sentiment news) | M3 - M2 (regime input) |
|---|---|---|---|
| All test | **-0.050 [-0.087, -0.012]** | +0.016 [-0.017, 0.049] | +0.017 [-0.002, 0.038] |
| Low-vol | -0.023 [-0.079, 0.031] | +0.006 [-0.041, 0.058] | +0.021 [-0.002, 0.043] |
| Mid-vol | -0.069 [-0.147, 0.003] | **+0.054 [0.004, 0.107]** | +0.021 [-0.011, 0.051] |
| High-vol | +0.005 [-0.062, 0.074] | +0.020 [-0.041, 0.081] | -0.043 [-0.101, 0.011] |
| Stress window | -0.024 [-0.115, 0.056] | -0.001 [-0.079, 0.072] | +0.002 [-0.044, 0.046] |
| Outside stress | **-0.060 [-0.097, -0.017]** | +0.016 [-0.021, 0.050] | **+0.024 [0.002, 0.047]** |

Also: M3 - M1 (news and regime versus price only) is -0.033 [-0.069, 0.001] overall.

Findings:
1. Adding the 768-dimensional FinBERT embedding to the price model reduced Macro-F1 by 0.050, and the interval excludes zero.
2. A low-capacity news input (three FinBERT sentiment probabilities) gave no reliable overall gain (+0.016, interval includes zero). The only interval excluding zero is in the Mid-vol regime (+0.054); this is one of 24 comparisons and is treated as a hypothesis.
3. Adding the regime label to the price+news model changed Macro-F1 by +0.017 overall (interval includes zero); it is nominally positive outside the stress window (+0.024) and not distinguishable from zero elsewhere. The best news-plus-regime model (M3) did not beat price only.
4. In the Feb-Apr 2020 stress window, no model with news information differs reliably from price only. Intervals are wide (about +/-0.08 to 0.10), so moderate effects cannot be excluded.

### 5.5 Hypotheses

| Hypothesis | Outcome |
|---|---|
| H1: looser protocols overstate performance | Not supported in general; supported for the regime-only model. |
| H2: the value of news differs across regimes | Not supported for embeddings (no gain in any regime). For sentiment, a gain appears only in Mid-vol and is not robust; insufficient evidence of regime dependence. |
| H3: an explicit regime input improves the multimodal model | Not supported (+0.017, interval includes zero; M3 does not beat M1). |

---

## 6. Discussion (draft)

**Class patterns and regimes.** Every learned model exceeds the uniform-random baseline on mean test Macro-F1. The regime-conditioned label distributions differ: in the training sample, Neutral accounts for about 64% of Low-vol rows, 55% of Mid-vol rows and 29% of High-vol rows. This is consistent with an association between recent volatility and the next-day class distribution. It does not prove that the models primarily predict volatility magnitude rather than direction. The price-only model is already competitive, and adding an explicit regime input to the price-plus-embedding model gives no reliable overall improvement. Per-class F1 (10 seeds, strict protocol) shows the same pattern: the Neutral class, which is where the volatility signal sits, is identical for the price-only and price+sentiment models (0.687 vs 0.690). The sentiment model trades Rise for Fall (Fall F1 0.291 to 0.397, Rise F1 0.420 to 0.358) and predicts Fall more often (32% of days vs 19% for price-only; true share 23%), which nets out to the small, statistically unreliable overall gain.

**Why the embedding hurts.** The 768-dimensional embedding must be learned from about 1,460 training rows. Under the strict protocol the news-only model reaches Macro-F1 0.701 on training rows but 0.382 on test, and adding the embedding to the price model raises training Macro-F1 from 0.477 to 0.554 while test Macro-F1 falls from 0.466 to 0.416; the price-only model shows almost no train/test gap (0.477 vs 0.466). This large train/test gap indicates overfitting, and the significant decrease when the embedding is added to the price model is consistent with that. News headlines also changed across the decade (average FinBERT sentiment -0.07 in train, -0.18 in test; 49 vs 26 headlines per day after capping), so a high-capacity news model may learn era-specific patterns that do not transfer.

**Why protocol effects are small.** The protocol comparison does not show systematic optimism across all models. The regime-only model scores higher under both looser protocols, which is consistent with slowly varying regime assignments being sensitive to random splitting and all-data clustering. However, because each protocol uses a different test sample, the comparison cannot attribute the full gap to leakage alone. This conclusion is limited to the specific protocol changes tested here.

**Stress period.** News was not more valuable during the Feb-Apr 2020 episode in this data, but with 50 rows the test has low power. [Connect to related work you actually read: e.g. studies that find sentiment more informative in volatile markets, and state whether your result agrees or whether your data are too limited to say.]

---

## 7. Limitations (draft)

- One asset (SPY), one benchmark and one decade; results may not transfer to other markets.
- 317 test rows (125/113/79 by regime, 50 in the stress window); intervals are wide and small effects cannot be detected.
- Headlines carry only a date, not a publication time. The label is the next trading day's return, so headlines dated t precede the predicted move provided they were published by the close of day t; this could not be verified.
- Headline volume and tone differ across the three periods, so train and test distributions differ.
- Regimes are data-driven clusters (3 clusters on rolling 20-day return and log-volatility), not validated economic states; the Mid-vol and High-vol regimes switch frequently (median run length 4 days).
- The three protocols were evaluated on different test samples, so gaps combine protocol changes and sample-composition differences; this design cannot isolate a causal effect of leakage alone. Only the specific leakage mechanisms implemented in P1/P2 were examined.
- The study examined 24 pairwise comparisons; four intervals exclude zero, so isolated findings should be treated as hypotheses.
- Prediction is not causation, and no trading strategy, costs or profitability are claimed.

---

## 8. Conclusion (draft)

On the NIFTY benchmark, with strict chronological evaluation, a small price-only LSTM was the model to beat. FinBERT embeddings of daily headlines reduced out-of-sample Macro-F1, a low-dimensional sentiment summary gave at most a small and statistically unreliable gain, and an explicit market-regime input added little. We found no evidence that news is more valuable during the 2020 stress period, though the sample is small. Moving from a strict to a looser evaluation protocol did not systematically inflate results, except for a model relying solely on a persistent regime label. These findings show that, on this benchmark, the tested FinBERT embedding did not improve on price-only prediction, the sentiment input's apparent gain was statistically uncertain, and news did not show a reliable advantage in the small stress-window sample. The protocol comparison is also limited because P1, P2 and P3 use different test samples; therefore, the observed gaps combine protocol changes with sample-composition differences. Claims about news or regime information should be tested against price-only baselines and strict out-of-time evaluation.

---

## Appendix: transparency statement (include in the report)

- The primary models (M0-M5) and the three protocols were specified before running experiments. M0r (random guess) and M6 (price + sentiment) were added after the pilot run showed that the news-only and price+news embedding models overfit; results for both are reported regardless of outcome.
- The number of seeds was increased from 3 to 10 after observing that seed-to-seed variation was as large as the effects of interest. No hyperparameters were changed.
- Price inputs were converted to scale-free features (returns, ratios, bounded indicators) before any model was run, because raw price levels drift over the sample.
- Up to 50 headlines per day were sampled at random (fixed seed) to bound compute and reduce volume differences across periods.
- 15 early rows without 20 days of history were excluded so that all models use the same samples (1,462 / 317 / 317 for train / validation / test).
- The label alignment (next-trading-day close-to-close return) was verified empirically on 2,110 rows.
