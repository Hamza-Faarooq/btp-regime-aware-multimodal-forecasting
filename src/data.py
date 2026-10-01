"""Step 1: NIFTY dataset loading, cleaning, lagged windows, and timing audit."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATASET_ID = "raeidsaqur/NIFTY"

MARKET_COLUMNS = [
    "open", "high", "low", "close", "adj_close", "volume", "pct_change",
    "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
]

LABEL_ORDER = ["Fall", "Neutral", "Rise"]


def load_nifty(dataset_id: str = DATASET_ID) -> pd.DataFrame:
    """Download the three official Hugging Face splits and combine them."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    frames = []
    for split_name in ("train", "valid", "test"):
        frame = ds[split_name].to_pandas()
        frame["split"] = {"train": "train", "valid": "val", "test": "test"}[split_name]
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def parse_context(context: str) -> dict:
    """Parse the comma-separated market context stored in each NIFTY row."""
    row = next(csv.DictReader(StringIO(context)))
    out = {}
    for key, value in row.items():
        key = key.strip()
        value = value.strip() if isinstance(value, str) else value
        if key == "date":
            out[key] = pd.to_datetime(value).normalize()
        else:
            out[key] = pd.to_numeric(value, errors="coerce")
    return out


def parse_market_context(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame(df["context"].map(parse_context).tolist())
    parsed["date"] = pd.to_datetime(parsed["date"]).dt.normalize()
    for col in MARKET_COLUMNS:
        if col in parsed:
            parsed[col] = pd.to_numeric(parsed[col], errors="coerce")
    return parsed


def parse_headlines(news_value) -> list[str]:
    """Convert the dataset news field to a clean per-day list."""
    if news_value is None or (isinstance(news_value, float) and np.isnan(news_value)):
        return []
    if isinstance(news_value, list):
        return [str(x).strip() for x in news_value if str(x).strip()]
    text = str(news_value).strip()
    return [text] if text else []


def build_clean_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Build one chronological table with market data, headlines, labels and returns."""
    market = parse_market_context(raw)

    clean = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]).dt.normalize(),
        "split": raw["split"].astype(str),
        "news": raw["news"].map(parse_headlines),
        "label": raw["label"].astype(str),
        "daily_return": pd.to_numeric(raw["pct_change"], errors="coerce"),
    })

    for col in MARKET_COLUMNS:
        if col in market.columns:
            clean[col] = market[col].values

    clean = clean.sort_values("date").reset_index(drop=True)

    if clean["date"].duplicated().any():
        dupes = clean.loc[clean["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Duplicate dates found: {dupes[:5]}")

    if not clean["date"].equals(market["date"]):
        raise ValueError("Dataset date and context date disagree.")

    return clean


def date_gap_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return the gaps between consecutive dataset dates."""
    dates = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)
    return pd.DataFrame({"date": dates, "gap_days": dates.diff().dt.days})


def build_lagged_windows(
    df: pd.DataFrame,
    lookback: int = 10,
    feature_columns: Iterable[str] = MARKET_COLUMNS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create target-day rows using only the previous lookback dataset rows.

    This uses row-based windows because the Step-1 brief permits option (b)
    when the supplied dataset has missing trading dates. The report must state
    this choice explicitly.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    feature_columns = list(feature_columns)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    windows, meta = [], []
    for target_idx in range(lookback, len(df)):
        past = df.iloc[target_idx - lookback:target_idx]
        target = df.iloc[target_idx]
        windows.append(past[feature_columns].to_numpy(dtype=np.float32))
        meta.append({
            "target_date": target["date"],
            "latest_input_date": past["date"].iloc[-1],
            "split": target["split"],
            "label": target["label"],
            "daily_return": target["daily_return"],
            "news": target["news"],
        })

    return np.stack(windows).astype(np.float32), pd.DataFrame(meta)


def timing_sanity_check(
    windows: np.ndarray,
    meta: pd.DataFrame,
    n_examples: int = 5,
) -> None:
    """Print examples and assert every input window ends before its target."""
    print("Timing sanity check")
    print("=" * 80)
    print(meta[["target_date", "latest_input_date", "split", "label"]].head(n_examples).to_string(index=False))

    violations = meta["latest_input_date"] >= meta["target_date"]
    if violations.any():
        bad = meta.loc[violations, ["target_date", "latest_input_date"]].head()
        raise AssertionError(f"Look-ahead detected:\n{bad}")

    assert (meta["latest_input_date"] < meta["target_date"]).all()
    print(f"PASS: {len(meta):,} windows have latest_input_date < target_date.")


def save_clean_table(df: pd.DataFrame, path: str = "data/clean.parquet") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
