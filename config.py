# BTP project configuration
# Fixed before experimentation; do not tune on the test set.

SEEDS = [0, 1, 2]
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
LOOKBACK = 10
LSTM_HIDDEN = 64
DROPOUT = 0.2
NEWS_DIM = 768
NEWS_HIDDEN = 64
N_REGIMES = 3
REGIME_EMBED_DIM = 8
FUSION_HIDDEN = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
