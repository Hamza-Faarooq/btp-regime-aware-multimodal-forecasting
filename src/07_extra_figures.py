"""Three extra results figures drawn only from numbers already saved in results/ (no model is re-run):
   fig_seed_scatter.png      all 240 per-run test Macro-F1 values (results/logs/04_models_grid_all_protocols.txt)
   fig_train_valid_test.png  P3 train / validation / test Macro-F1 (results/EXECUTED_RESULTS.md, section 2-3)
   fig_class_shares.png      P3 predicted vs true class shares (results/logs/06_class_balance_diagnostic.txt)
    python src/07_extra_figures.py
"""
import os, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("BTP_ROOT", Path(__file__).resolve().parents[1])); RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAME = {"M0": "M0\nmajority", "M0r": "M0r\nrandom", "M1": "M1\nprice", "M2": "M2\nprice+\nFinBERT", "M3": "M3\n+ regime", "M4": "M4\nnews only", "M5": "M5\nregime only", "M6": "M6\nprice+\nsentiment"}
MODELS = list(NAME); PCOL = {"P1": "#c44e52", "P2": "#dd8452", "P3": "#4c72b0"}

# ---- 1. per-run scatter (240 runs)
runs = {}
for line in open(RES / "logs" / "04_models_grid_all_protocols.txt"):
    m = re.match(r"(P\d) (M0r?|M\d) seed(\d+): test macro-F1 ([0-9.]+)", line)
    if m: runs.setdefault((m[1], m[2]), []).append(float(m[4]))
assert sum(len(v) for v in runs.values()) == 240, sum(len(v) for v in runs.values())
fig, ax = plt.subplots(figsize=(8.2, 4.2)); rng = np.random.default_rng(0)
for j, m in enumerate(MODELS):
    for k, p in enumerate(["P1", "P2", "P3"]):
        v = np.array(runs[(p, m)]); x = j + (k - 1) * 0.26
        ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=14, color=PCOL[p], alpha=.75, label=f"{p}" if j == 0 else None, zorder=3)
        ax.hlines(v.mean(), x - 0.11, x + 0.11, color="k", lw=1.6, zorder=4)
ax.axhline(0.333, color="grey", ls=":", lw=1); ax.text(7.45, 0.338, "0.333", fontsize=8, color="grey", ha="right")
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=8.5)
ax.set_ylabel("Test Macro-F1 (one dot per seed)"); ax.set_title("All 240 evaluations: seed-to-seed spread (black bar = mean)", fontsize=11)
ax.legend(title="Protocol", fontsize=9, ncol=3, loc="lower right"); plt.tight_layout(); fig.savefig(FIG / "fig_seed_scatter.png", dpi=200); plt.close(fig)

# ---- 2. P3 train / validation / test Macro-F1
txt = open(RES / "EXECUTED_RESULTS.md").read()
def block(tag, nxt):
    i = txt.index(tag); seg = txt[i:]; seg = seg[:seg.index(nxt)] if nxt in seg else seg
    out = {}
    for l in seg.splitlines():
        t = l.split()
        if len(t) >= 3 and t[0] in MODELS: out[t[0]] = (float(t[1]), float(t[2]))
    return out
sec2 = txt[txt.index("## 2."):txt.index("## 3.")]; test = {}
for l in sec2.splitlines()[sec2.splitlines().index(next(x for x in sec2.splitlines() if x.startswith("P3 TEST") or "P3" in x and "TEST" in x)) if any(x.startswith("P3 TEST") for x in sec2.splitlines()) else 0:]:
    t = l.split()
    if len(t) >= 3 and t[0] in MODELS and t[0] not in test: test[t[0]] = (float(t[1]), float(t[2]))
valid = block("P3 VALID", "P3 TRAIN"); train = block("P3 TRAIN", "```")
assert all(len(d) == 8 for d in (test, valid, train)), (len(test), len(valid), len(train))
fig, ax = plt.subplots(figsize=(8.2, 4.0)); w = 0.26
for k, (lab, d, c) in enumerate([("Train", train, "#9db4d8"), ("Validation", valid, "#e5c45c"), ("Test", test, "#c44e52")]):
    ax.bar(np.arange(8) + (k - 1) * w, [d[m][0] for m in MODELS], w, yerr=[d[m][1] for m in MODELS], capsize=2, color=c, label=lab)
ax.axhline(0.333, color="grey", ls=":", lw=1)
ax.set_xticks(range(8)); ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=8.5); ax.set_ylim(0.2, None); ax.set_ylabel("Macro-F1 (mean ± s.d., 10 seeds)")
ax.set_title("Strict protocol (P3): fit to train versus performance on new data", fontsize=11); ax.legend(fontsize=9, ncol=3, loc="upper left")
plt.tight_layout(); fig.savefig(FIG / "fig_train_valid_test.png", dpi=200); plt.close(fig)

# ---- 3. predicted vs true class shares
sh = {}
for l in open(RES / "logs" / "06_class_balance_diagnostic.txt"):
    m = re.match(r"(M\d) predicted share Fall/Neutral/Rise: \[([0-9. ]+)\]", l)
    if m: sh[m[1]] = [float(x) for x in m[2].split()]
    m = re.match(r"true test share: \[([0-9. ]+)\]", l)
    if m: sh["True"] = [float(x) for x in m[1].split()]
order = ["True", "M1", "M2", "M3", "M6"]; assert all(k in sh for k in order)
lab = {"True": "True share", "M1": "M1 price", "M2": "M2 price+FinBERT", "M3": "M3 +regime", "M6": "M6 price+sentiment"}
cols = ["#555555", "#4c72b0", "#dd8452", "#55a868", "#8172b3"]; fig, ax = plt.subplots(figsize=(8.2, 4.0)); w = 0.16
for k, key in enumerate(order):
    ax.bar(np.arange(3) + (k - 2) * w, sh[key], w, color=cols[k], label=lab[key])
    for i, v in enumerate(sh[key]): ax.text(i + (k - 2) * w, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["Fall", "Neutral", "Rise"]); ax.set_ylabel("Share of the 317 test rows"); ax.set_ylim(0, 0.76)
ax.set_title("Predicted class shares under P3 (mean over 10 seeds) versus the true shares", fontsize=11); ax.legend(fontsize=8.5, ncol=3, loc="upper left")
plt.tight_layout(); fig.savefig(FIG / "fig_class_shares.png", dpi=200); plt.close(fig)
print("wrote fig_seed_scatter.png, fig_train_valid_test.png, fig_class_shares.png")
