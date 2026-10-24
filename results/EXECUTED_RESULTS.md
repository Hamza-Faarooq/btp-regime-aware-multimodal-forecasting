# Executed results (10 seeds, LAG = 0, M0-M6)

Everything below is copied from the raw cell outputs of `notebooks/BTP_pipeline_executed.ipynb` (also saved as text in `results/logs/`).
Nothing is re-typed or rounded further. M7 (gated fusion) and the LAG = 1 run were **not** executed on real data.

> **Reading notes (added after the consistency audit; raw outputs below are untouched).**
> * The "15 missing / expected 2,689 trading days" line in the step-1 audit log comes from a US-federal-holiday approximation. Against the NYSE calendar only 2018-12-31 is absent; the other 14 listed dates are market closures.
> * The seed-bootstrap intervals use 2,000 resamples (`seed_gap`, n=2000); the row-bootstrap intervals use 1,000.
> * "seed-averaged" in the printed headings means: the mean over the ten seeds of the per-seed Macro-F1 (predictions are never averaged across seeds). The bootstrap resamples test rows and recomputes each seed's Macro-F1 on every resample.
> * 240 evaluations = 8 models x 3 protocols x 10 seeds, of which 180 are trained networks (M1-M6) and 60 are non-trained baselines (M0, M0r).

Conventions: Macro-F1 is the primary metric. Seeds 0-9. P3 = strict chronological protocol. Class order Fall / Neutral / Rise.
Test split: 317 rows, 2019-02-13 to 2020-09-21; true class shares 0.230 / 0.451 / 0.319.

## 1. All protocols, test split (mean / std over 10 seeds)

```
TEST Macro-F1 by protocol (mean/std over seeds):
            mean                  std              
protocol     P1     P2     P3     P1     P2     P3
model                                             
M0        0.228  0.207  0.207  0.006  0.000  0.000
M0r       0.315  0.333  0.333  0.018  0.020  0.020
M1        0.438  0.460  0.466  0.034  0.036  0.036
M2        0.423  0.428  0.416  0.014  0.025  0.023
M3        0.421  0.441  0.433  0.021  0.020  0.025
M4        0.363  0.369  0.382  0.010  0.040  0.037
M5        0.400  0.382  0.354  0.022  0.030  0.019
M6        0.431  0.448  0.482  0.035  0.051  0.038
```

## 2. P3 test, accuracy and balanced accuracy

```

       macro_f1        accuracy        balanced_acc       
          mean    std     mean    std         mean    std
model                                                    
M0       0.207  0.000    0.451  0.000        0.333  0.000
M0r      0.333  0.020    0.339  0.019        0.341  0.021
M1       0.466  0.036    0.530  0.016        0.476  0.031
M2       0.416  0.023    0.493  0.030        0.448  0.018
M3       0.433  0.025    0.491  0.024        0.454  0.016
M4       0.382  0.037    0.402  0.041        0.394  0.028
M5       0.354  0.019    0.493  0.032        0.415  0.006
M6       0.482  0.038    0.531  0.022        0.498  0.019
```

## 3. P3 validation and training (train/test gap shows overfitting of the embedding models)

```
P3 VALID
       macro_f1        accuracy        balanced_acc       
          mean    std     mean    std         mean    std
model                                                    
M0       0.249  0.000    0.596  0.000        0.333  0.000
M0r      0.291  0.021    0.317  0.024        0.311  0.023
M1       0.464  0.018    0.570  0.007        0.469  0.016
M2       0.452  0.015    0.600  0.019        0.450  0.014
M3       0.461  0.009    0.600  0.014        0.458  0.007
M4       0.378  0.021    0.536  0.038        0.389  0.020
M5       0.390  0.014    0.605  0.015        0.436  0.016
M6       0.456  0.008    0.561  0.009        0.459  0.009
P3 TRAIN
       macro_f1        accuracy        balanced_acc       
          mean    std     mean    std         mean    std
model                                                    
M0       0.232  0.000    0.535  0.000        0.333  0.000
M0r      0.315  0.016    0.330  0.016        0.328  0.018
M1       0.477  0.053    0.531  0.038        0.483  0.058
M2       0.554  0.051    0.599  0.032        0.562  0.058
M3       0.569  0.071    0.607  0.053        0.585  0.076
M4       0.701  0.130    0.709  0.125        0.731  0.130
M5       0.360  0.017    0.537  0.034        0.406  0.004
M6       0.486  0.052    0.539  0.040        0.491  0.056
```

## 4. Protocol gaps and news value by regime (95% bootstrap CIs)

```
Optimistic performance gap (positive = looser protocol scores higher):
    model      gap    est     lo     hi
0     M1  P1 - P3 -0.028 -0.055  0.001
1     M1  P2 - P3 -0.005 -0.034  0.023
2     M2  P1 - P3  0.006 -0.010  0.022
3     M2  P2 - P3  0.012 -0.008  0.031
4     M3  P1 - P3 -0.012 -0.030  0.008
5     M3  P2 - P3  0.008 -0.010  0.028
6     M4  P1 - P3 -0.019 -0.040  0.004
7     M4  P2 - P3 -0.013 -0.043  0.018
8     M5  P1 - P3  0.046  0.029  0.063
9     M5  P2 - P3  0.028  0.007  0.048
10    M6  P1 - P3 -0.051 -0.079 -0.019
11    M6  P2 - P3 -0.034 -0.071  0.005
```

```
Macro-F1 levels with 95% bootstrap CI (P3 test; seed-averaged):
model                                   M0r     M1     M2     M3     M4     M5     M6
group                                                                                
All test                              0.333  0.466  0.416  0.433  0.382  0.354  0.482
Regime: Low-vol                       0.329  0.409  0.386  0.407  0.396  0.246  0.415
Regime: Mid-vol                       0.322  0.410  0.341  0.361  0.339  0.209  0.464
Regime: High-vol                      0.323  0.357  0.363  0.319  0.342  0.189  0.378
Stress window 2020-02-01..2020-04-30  0.316  0.473  0.448  0.450  0.347  0.330  0.471
Outside stress window                 0.334  0.451  0.391  0.416  0.377  0.342  0.467

Differences (positive = second model better), 95% CI from resampling test rows:
                               group   n                   comparison    est               ci
                            All test 317  M2-M1 (embedding news gain) -0.050 [-0.087, -0.012]
                            All test 317  M6-M1 (sentiment news gain)  0.016  [-0.017, 0.049]
                            All test 317        M3-M2 (regime effect)  0.017  [-0.002, 0.038]
                            All test 317 M3-M1 (news+regime vs price) -0.033  [-0.069, 0.001]
                     Regime: Low-vol 125  M2-M1 (embedding news gain) -0.023  [-0.079, 0.031]
                     Regime: Low-vol 125  M6-M1 (sentiment news gain)  0.006  [-0.041, 0.058]
                     Regime: Low-vol 125        M3-M2 (regime effect)  0.021  [-0.002, 0.043]
                     Regime: Low-vol 125 M3-M1 (news+regime vs price) -0.002  [-0.067, 0.065]
                     Regime: Mid-vol 113  M2-M1 (embedding news gain) -0.069  [-0.147, 0.003]
                     Regime: Mid-vol 113  M6-M1 (sentiment news gain)  0.054   [0.004, 0.107]
                     Regime: Mid-vol 113        M3-M2 (regime effect)  0.021  [-0.011, 0.051]
                     Regime: Mid-vol 113 M3-M1 (news+regime vs price) -0.049  [-0.116, 0.022]
                    Regime: High-vol  79  M2-M1 (embedding news gain)  0.005  [-0.062, 0.074]
                    Regime: High-vol  79  M6-M1 (sentiment news gain)  0.020  [-0.041, 0.081]
                    Regime: High-vol  79        M3-M2 (regime effect) -0.043  [-0.101, 0.011]
                    Regime: High-vol  79 M3-M1 (news+regime vs price) -0.038  [-0.102, 0.021]
Stress window 2020-02-01..2020-04-30  50  M2-M1 (embedding news gain) -0.024  [-0.115, 0.056]
Stress window 2020-02-01..2020-04-30  50  M6-M1 (sentiment news gain) -0.001  [-0.079, 0.072]
Stress window 2020-02-01..2020-04-30  50        M3-M2 (regime effect)  0.002  [-0.044, 0.046]
Stress window 2020-02-01..2020-04-30  50 M3-M1 (news+regime vs price) -0.022   [-0.11, 0.061]
               Outside stress window 267  M2-M1 (embedding news gain) -0.060 [-0.097, -0.017]
               Outside stress window 267  M6-M1 (sentiment news gain)  0.016   [-0.021, 0.05]
               Outside stress window 267        M3-M2 (regime effect)  0.024   [0.002, 0.047]
               Outside stress window 267 M3-M1 (news+regime vs price) -0.036  [-0.073, 0.006]
```

## 5. Class-balance diagnostic (why Macro-F1 and not accuracy)

```
M1 predicted share Fall/Neutral/Rise: [0.191 0.545 0.263]
M2 predicted share Fall/Neutral/Rise: [0.303 0.581 0.116]
M3 predicted share Fall/Neutral/Rise: [0.302 0.525 0.173]
M6 predicted share Fall/Neutral/Rise: [0.322 0.502 0.176]
true test share: [0.23  0.451 0.319]
M1 per-class F1 (Fall/Neutral/Rise): [0.291 0.687 0.42 ]
M6 per-class F1 (Fall/Neutral/Rise): [0.397 0.69  0.358]
```

## 6. Regimes (P3 fit, training rows) and persistence

```
(2096, 10) (2695, 17)
NaNs in regime features: 0
share of rows with the same regime under train-only vs all-data fit: 0.9356
{'test': 0.9495, 'train': 0.9227, 'valid': 0.9811}
Regime stats on TRAIN rows (P3 fit):
             n  mean_ret20  mean_vol20  mean_next_day_ret   share
Low-vol   700      0.0185      0.0055            -0.0000  0.4788
Mid-vol   411      0.0319      0.0103             0.0009  0.2811
High-vol  351     -0.0363      0.0126             0.0010  0.2401

Regime share by split (P3 fit):
 regime_p3  High-vol  Low-vol  Mid-vol
split                                
test          0.249    0.394    0.356
train         0.240    0.479    0.281
valid         0.243    0.640    0.117

Label mix within each regime (P3 fit), all splits:
 label       Fall  Neutral   Rise
regime_p3                       
High-vol   0.306    0.288  0.406
Low-vol    0.164    0.642  0.194
Mid-vol    0.182    0.549  0.269
Transition matrix (daily, P3 model):
           Low-vol  Mid-vol  High-vol
Low-vol     0.961    0.017     0.021
Mid-vol     0.058    0.869     0.073
High-vol    0.016    0.109     0.875

Run length in trading days:
           count  mean  median  max
regime                            
High-vol     81   7.9     4.0   39
Low-vol      51  25.9    15.0  333
Mid-vol      93   7.7     4.0   40
saved regimes.parquet, stats, transition matrix, models
```

## How to read these numbers

* About 24 comparisons are shown. A single interval that excludes zero (for example M6-M1 in Mid-vol, +0.054 [0.004, 0.107]) is a hypothesis, not a finding.
* Test rows are 317 overlapping, autocorrelated days, so bootstrap intervals are probably somewhat too narrow.
* The high-dimensional news embedding (M2) is worse than price only (M1): -0.050 [-0.087, -0.012], while its training score is far higher (0.554 vs 0.477). That is overfitting, not a bug.
* The compact sentiment model M6 is slightly higher than M1 on average, but M6-M1 = +0.016 [-0.017, 0.049] includes zero. M6 was added after the first results were seen (see README, "Transparency").
* Protocol gaps are small and model-dependent; only M5 (positive) and M6 (negative) have intervals excluding zero. The study therefore does **not** find that loose protocols uniformly inflate scores on this dataset.

## Machine-readable tables

`audit_gaps.csv`, `p3_levels_by_group.csv` and `p3_diffs_by_group.csv` in this folder were **parsed from the printed notebook output** (`results/logs/`), so they hold exactly the rounded values shown above. The original unrounded CSVs, per-seed metrics and prediction files live in the author's Google Drive `btp/results/`.
