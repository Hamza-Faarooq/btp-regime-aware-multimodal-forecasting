# BTP project configuration -- REFERENCE COPY of the frozen settings.
# The scripts in src/ define these values inline (they are not imported from here); keep this file in sync if anything changes.
# Frozen before final experimentation; never tuned on the test set.

SEEDS = list(range(10))
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

LOOKBACK = 10
LAG = 0
LSTM_HIDDEN = 64
DROPOUT = 0.2
NEWS_DIM = 768
NEWS_HIDDEN = 64
REGIME_EMBED_DIM = 8
FUSION_HIDDEN = 64

N_REGIMES = 3
REGIME_WINDOW = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
CLIP = 5.0

NEWS_CAP = 50
NEWS_SAMPLE_SEED = 1234
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LEN = 64
FINBERT_BATCH_SIZE = 256

BOOTSTRAP_RESAMPLES = 1000        # row bootstrap for model differences (notebook NBOOT)
SEED_BOOTSTRAP_RESAMPLES = 2000   # bootstrap over per-seed scores for protocol gaps (notebook seed_gap n)
STRESS_START = "2020-02-01"
STRESS_END = "2020-04-30"
REGIME_NAMES = ["Low-vol", "Mid-vol", "High-vol"]
