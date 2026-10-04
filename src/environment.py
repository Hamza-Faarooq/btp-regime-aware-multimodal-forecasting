# Step 0 — Environment & Project Setup
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

# ===== Original notebook cell 0 =====
from google.colab import drive
drive.mount('/content/drive')

import os
ROOT = '/content/drive/MyDrive/btp'
for d in ['data','src','results','figures','notebooks','report','presentation','literature']:
    os.makedirs(f'{ROOT}/{d}', exist_ok=True)
os.chdir(ROOT)
print(os.listdir(ROOT))
