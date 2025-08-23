# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: hello-cursor-jupyter-notebook-pytorch
#     language: python
#     name: hello-cursor-jupyter-notebook-pytorch
# ---

# %% [markdown]
# # imports

# %%
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# %% [markdown]
# # constants

# %%
BATCH_SIZE = 32
CSV_FILE = '../data/mpg-pounds.csv'

# %% [markdown]
# # fetch


# %%
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.pounds = torch.tensor(df["pounds"].values, dtype=torch.float32)
        self.mpg = torch.tensor(df["mpg"].values, dtype=torch.float32)
        
    def __len__(self):
        return self.pounds.shape[0]
    
    def __getitem__(self, idx):
        return self.pounds[idx].view(-1, 1), self.mpg[idx].view(-1, 1)

dataset = CustomDataset(CSV_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")
print(f"Batch size: {BATCH_SIZE}")
