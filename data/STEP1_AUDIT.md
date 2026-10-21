# Step 1 audit: what the data actually is

All numbers come from `results/logs/01_data_load_and_timing_audit.txt` and `01b_eda.txt` (reproduce with `src/01_data.py`).

## Dataset
* NIFTY (`raeidsaqur/NIFTY` on Hugging Face), market = SPY. 2,111 rows: train 1,477 (2010-01-06 to 2017-06-27), validation 317 (to 2019-02-12), test 317 (2019-02-13 to 2020-09-21). Official splits are kept.
* Each row has a text prompt with a few days of market context (4-8 trading days; most rows 5-7), a block of headlines, a label and `pct_change`.

## Findings that shaped the design
1. **The label is the NEXT trading day's close-to-close return.** Verified on 2,110 rows: share within 1e-3 of the next-day return = 1.000, versus 0.064-0.087 for same-day, previous-day or open-close alternatives. Labels follow Fall < -0.5%, Neutral, Rise > +0.5%; 10 of 2,111 rows sit exactly on the +/-0.5% boundary and disagree by rounding (listed in the log).
2. **The dataset is not a daily series** (target-date gaps of 1 to 4 business days; 2,111 rows vs about 2,689 trading days), but the overlapping contexts let us recover **2,695 distinct trading days** (2010-01-05 to 2020-09-18; about 15 expected days are absent, e.g. some Good Fridays and 2012-10-29/30, 2018-12-05). Duplicate days agree exactly. Windows are built from this recovered daily series, so "10 days" means 10 trading days (max calendar span 18 days).
3. **Timing.** A window ends at date t (LAG = 0) and predicts day t+1's move. The window may therefore contain day t's price and day t's headlines, both known before the predicted move. Assertions in the code check that no window or regime feature uses a day after its target. `LAG = 1` (benchmark's own t-1 convention) is implemented as a robustness option but was not run on real data.
4. **Warm-up.** 15 earliest rows lack 20 days of history for the regime features and are dropped: train 1,462, validation 317, test 317 (2,096 rows).
5. **Class balance shifts across splits** (share Fall / Neutral / Rise): train 0.205 / 0.535 / 0.260; validation 0.167 / 0.596 / 0.237; test 0.230 / 0.451 / 0.319. Always predicting the majority gives Macro-F1 of only 0.207 on test, so a uniform random guess (0.333) is used as the real floor.
6. **Headline volume changes by era.** Mean headlines per day: train 277 (max 1,627), validation 54, test 26. FinBERT embeddings are built from at most 50 randomly sampled headlines per day (fixed seed), but the changing news supply is a confound for any news result and is stated as a limitation.

## Leakage guard
For every window and regime feature the code asserts `latest_input_date <= target_date - 1 day` (LAG = 0 uses day t as the last input for the t+1 label; see item 3) and that all splits are chronological.
