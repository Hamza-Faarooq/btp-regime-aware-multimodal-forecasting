# Step 1 audit note

## Verified from the NIFTY dataset paper and Hugging Face dataset card

- Dataset: raeidsaqur/NIFTY, NIFTY-LM split used here.
- The market target is the SPY index.
- Total records: 2,111.
- Official splits: train 1,477, validation 317, test 317.
- Date ranges:
  - train: 2010-01-06 to 2017-06-27
  - validation: 2017-06-28 to 2019-02-12
  - test: 2019-02-13 to 2020-09-21
- The supplied market context contains historical market statistics; the dataset paper describes the context as approximately the previous 10 days.
- The dataset does not contain every calendar date; the paper attributes missing dates to the trading calendar and absence of relevant news headlines.
- Labels use the supplied percentage change:
  - Fall: < -0.5%
  - Neutral: -0.5% to +0.5%, inclusive
  - Rise: > +0.5%

## Implementation decision

Because the supplied dataset is not a complete daily sequence, Step 1 uses **option (b): row-based windows**. A target at dataset row t uses the previous 10 available dataset rows (t-10 through t-1), rather than inventing missing market observations.

This is an implementation choice for this BTP and must be stated as a limitation: the model's "10-day" history means 10 consecutive observations in the supplied dataset, not necessarily 10 consecutive exchange trading sessions.

## Leakage guard

For every constructed window, the code records target_date and latest_input_date and asserts:

latest_input_date < target_date

The target day's market fields, including its close/high/low and supplied pct_change, are never placed in its own market window. Historical pct_change values from prior rows are allowed as lagged market features.

## Runtime checkpoint still required

The Colab run must provide the actual date-gap frequencies, missing-value counts, class counts, headline counts, generated Parquet file, plots, and a passing timing assertion. These runtime outputs are deliberately not fabricated in the repository before the dataset is executed.
