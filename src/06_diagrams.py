"""Draws the three design diagrams used in the report and slides (no data needed).

    python src/06_diagrams.py            # writes figures/fig_architecture.png, fig_protocol_design.png, fig_dataset_timeline.png

Facts drawn here come from the executed pipeline: window = 10 trading days x 12 scale-free features, <=50 sampled headlines per
day, frozen ProsusAI/finbert, LSTM(64), news MLP 768->64, regime embedding 3->8, head 64->3, KMeans k=3, official NIFTY split.
"""
import os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(os.environ.get("BTP_ROOT", Path(__file__).resolve().parents[1])); FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
C = dict(price="#4c72b0", news="#dd8452", regime="#55a868", fuse="#8172b3", out="#c44e52", ink="#222222", grey="#9aa5b1", light="#f4f6f8")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

def box(ax, x, y, w, h, text, color, fc=None, bold=False, fs=9.5, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2", fc=fc or color + "26", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or C["ink"], fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(ax, x0, y0, x1, y1, color=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.5, color=color or C["ink"], connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

# ---------------------------------------------------------------- 1. architecture
# Drawn at the size it is printed (about 8.4 in wide) so that text stays >= ~7 pt on the page; explanatory notes are in the report caption.
fig, ax = plt.subplots(figsize=(8.4, 5.0)); ax.set_xlim(0, 100); ax.set_ylim(4, 58); ax.axis("off")
ax.text(50, 56, "Model family: one network, switchable inputs", ha="center", fontsize=12, fontweight="bold")
for x, t in [(12, "Inputs"), (42, "Encoders"), (70, "Fusion"), (91.5, "Output")]:
    ax.text(x, 51.5, t, ha="center", fontsize=9.5, color=C["grey"], fontweight="bold")
rows = {"price": 37, "news": 22, "regime": 7}
box(ax, 0.5, rows["price"], 23, 11, "Price window\n10 days x 12\nscale-free features", C["price"], fs=8.8)
box(ax, 0.5, rows["news"], 23, 11, "Daily headlines\n(up to 50 per day,\nfixed seed)", C["news"], fs=8.8)
box(ax, 0.5, rows["regime"], 23, 11, "Regime features\n20-day return and\n20-day volatility", C["regime"], fs=8.8)
box(ax, 29, rows["price"], 26, 11, "LSTM (64 units)\nlast hidden state\ndropout 0.2", C["price"], fs=8.8)
box(ax, 29, rows["news"], 26, 11, "Frozen FinBERT\n768-d mean-pool ->\nLinear 768->64, ReLU", C["news"], fs=8.8)
box(ax, 29, rows["regime"], 26, 11, "KMeans (k = 3)\nembedding 3 -> 8", C["regime"], fs=8.8)
for k in rows: arrow(ax, 24.6, rows[k] + 5.5, 28.2, rows[k] + 5.5, C[k])
box(ax, 61, 15, 17, 26, "Concatenate\nactive branches\n\nM1 price\nM2 + FinBERT\nM3 + regime\nM4 news only\nM5 regime only\nM6 + sentiment", C["fuse"], fs=8.6)
for k in rows: arrow(ax, 56.2, rows[k] + 5.5, 60.2, 28 + (rows[k] - 22) * 0.35, C[k])
box(ax, 84, 29, 15.5, 11, "MLP head\n64 -> 3", C["fuse"], fs=9)
arrow(ax, 79.2, 34.5, 83.2, 34.5, C["fuse"])
box(ax, 84, 11, 15.5, 12, "Fall / Neutral\n/ Rise\n(+/-0.5% bands)", C["out"], fs=8.6)
arrow(ax, 91.7, 28.2, 91.7, 24.0, C["out"])
fig.savefig(FIG / "fig_architecture.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- 2. protocol design
fig, ax = plt.subplots(figsize=(13.5, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
ax.text(50, 42, "Three evaluation protocols applied to the same models", ha="center", fontsize=14, fontweight="bold")
specs = [("P1  random split\n(deliberately invalid)", C["out"], "random", "all rows", "Overlapping windows and persistent regimes\nput near-copies of test rows in training."),
         ("P2  chronological split,\npreprocessing on all data", "#dd8452", "chrono", "all rows", "Temporal order respected, but scalers and\nregime clusters use test-period data."),
         ("P3  chronological split,\ntrain-only preprocessing (strict)", C["price"], "chrono", "train rows only", "Primary benchmark: nothing from validation or test\nshapes scalers or clusters; validation only selects\nthe early-stopping checkpoint.")]
for j, (title, col, split, fit, note) in enumerate(specs):
    x0 = 2 + j * 33
    ax.text(x0 + 14, 37, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col, linespacing=1.3)
    ax.text(x0, 31.2, "Rows / split", fontsize=8.6, color=C["grey"])
    n = 36
    import numpy as np
    rng = np.random.default_rng(3)
    if split == "chrono": lab = ["train"] * 24 + ["valid"] * 6 + ["test"] * 6
    else: lab = list(rng.permutation(["train"] * 24 + ["valid"] * 6 + ["test"] * 6))
    cc = {"train": C["price"], "valid": "#e5c45c", "test": C["out"]}
    for i, s in enumerate(lab): ax.add_patch(Rectangle((x0 + i * 0.78, 26), 0.72, 4, fc=cc[s], ec="none"))
    ax.text(x0, 23.6, "time ->", fontsize=8, color=C["grey"])
    ax.text(x0, 19.3, "Scalers and KMeans fitted on:", fontsize=8.6, color=C["grey"])
    for i in range(n):
        used = (fit == "all rows") or lab[i] == "train"
        ax.add_patch(Rectangle((x0 + i * 0.78, 14.5), 0.72, 3, fc=(C["ink"] if used else "#e6e9ed"), ec="none"))
    ax.text(x0, 11.6, fit, fontsize=9, fontweight="bold")
    ax.text(x0, 8.0, note, fontsize=8.6, va="top", color=C["ink"], linespacing=1.3)
for i, (s, c) in enumerate([("train", C["price"]), ("validation", "#e5c45c"), ("test", C["out"])]):
    ax.add_patch(Rectangle((34 + i * 12, 0.6), 1.6, 1.6, fc=c)); ax.text(36.2 + i * 12, 1.4, s, fontsize=8.6, va="center")
ax.text(5, 1.4, "Gaps between protocols:  P1 to P2 = ignoring time order    P2 to P3 = preprocessing / regime-fit leakage", fontsize=8.4, color=C["grey"], style="italic", va="center", ha="left") if False else None
fig.savefig(FIG / "fig_protocol_design.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- 3. dataset timeline
import datetime as dt
fig, ax = plt.subplots(figsize=(13.5, 3.9))
D = lambda s: dt.date.fromisoformat(s)
parts = [("Train\n1,462 rows after warm-up", "2010-01-06", "2017-06-27", C["price"]), ("Validation\n317 rows", "2017-06-28", "2019-02-12", "#e5c45c"), ("Test\n317 rows", "2019-02-13", "2020-09-21", C["out"])]
for name, a, b, col in parts:
    ax.barh(0, (D(b) - D(a)).days, left=D(a).toordinal(), height=0.5, color=col, alpha=0.9)
    mid = D(a).toordinal() + (D(b) - D(a)).days / (2 if not name.startswith("Test") else 3.2)
    ax.text(mid, 0, name, ha="center", va="center", color="white" if col != "#e5c45c" else C["ink"], fontsize=10, fontweight="bold", linespacing=1.3)
    ax.text(D(a).toordinal(), -0.42, a, fontsize=8, ha="left" if not name.startswith("Test") else "left", color=C["grey"]) if not name.startswith("Test") else None
ax.text(D("2020-09-21").toordinal(), -0.42, "2020-09-21", fontsize=8, ha="right", color=C["grey"]); ax.text(D("2019-02-13").toordinal(), -0.42, "2019-02-13", fontsize=8, ha="right", color=C["grey"])
s0, s1 = D("2020-02-01").toordinal(), D("2020-04-30").toordinal()
ax.add_patch(Rectangle((s0, -0.34), s1 - s0, 0.68, fc="none", ec=C["ink"], lw=1.6, ls="--"))
ax.annotate("Stress-period case study\n1 Feb - 30 Apr 2020 (50 test rows)", xy=((s0 + s1) / 2, 0.34), xytext=((s0 + s1) / 2 - 330, 0.88), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.2))
ax.set_ylim(-0.7, 1.25); ax.set_xlim(D("2009-10-01").toordinal(), D("2020-12-15").toordinal()); ax.axis("off")
ax.set_title("NIFTY / SPY benchmark: official chronological split", fontsize=13, fontweight="bold", loc="left")
fig.savefig(FIG / "fig_dataset_timeline.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote", [p.name for p in sorted(FIG.glob("fig_*.png"))])
