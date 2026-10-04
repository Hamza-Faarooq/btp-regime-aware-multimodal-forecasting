# Executed BTP Results

Source: attached working `BTP.ipynb`, final 10-seed run.

## Strict protocol P3 — test Macro-F1

| Model | Mean ± SD |
|---|---:|
| M0 | 0.207 ± 0.000 |
| M0r | 0.333 ± 0.020 |
| M1 | 0.466 ± 0.036 |
| M2 | 0.416 ± 0.023 |
| M3 | 0.433 ± 0.025 |
| M4 | 0.382 ± 0.037 |
| M5 | 0.354 ± 0.019 |
| M6 | 0.482 ± 0.038 |

## Protocol audit

| Model | P1 | P2 | P3 |
|---|---:|---:|---:|
| M1 | 0.438 | 0.460 | 0.466 |
| M2 | 0.423 | 0.428 | 0.416 |
| M3 | 0.421 | 0.441 | 0.433 |
| M4 | 0.363 | 0.369 | 0.382 |
| M5 | 0.400 | 0.382 | 0.354 |
| M6 | 0.431 | 0.448 | 0.482 |

Most looser-protocol gaps are small or negative. M5 is the clear positive exception.

## Strict-protocol group results

| Group | M1 | M2 | M3 | M6 |
|---|---:|---:|---:|---:|
| All test | 0.466 | 0.416 | 0.433 | 0.482 |
| Low-vol | 0.409 | 0.386 | 0.407 | 0.415 |
| Mid-vol | 0.410 | 0.341 | 0.361 | 0.464 |
| High-vol | 0.357 | 0.363 | 0.319 | 0.378 |
| Stress (Feb–Apr 2020) | 0.473 | 0.448 | 0.450 | 0.471 |
| Outside stress | 0.451 | 0.391 | 0.416 | 0.467 |

## Current interpretation

The 768-dimensional FinBERT embedding reduced Macro-F1 versus price-only by 0.050 overall. The 3-number sentiment channel improved the point estimate by 0.016 but its bootstrap interval includes zero. No news model showed a reliable advantage over price-only in the Feb–Apr 2020 stress window.

These are the results of the existing main experiment. The planned gated-fusion and LAG=1 robustness runs are not included here until executed.
