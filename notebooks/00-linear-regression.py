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
# # fetch

# %%
# Define a custom PyTorch dataset for the MPG data
class MPGDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file
        """
        self.data = pd.read_csv(csv_file)
        # Convert to PyTorch tensors
        self.pounds = torch.tensor(self.data['pounds'].values, dtype=torch.float32)
        self.mpg = torch.tensor(self.data['mpg'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.pounds[idx], self.mpg[idx]

# Load the dataset
dataset = MPGDataset('data/mpg-pounds.csv')

# Create a DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Display dataset info
print(f"Dataset size: {len(dataset)}")
print(f"First few samples:")
for i in range(min(5, len(dataset))):
    pounds, mpg = dataset[i]
    print(f"  Sample {i}: Weight = {pounds:.3f} pounds, MPG = {mpg:.2f}")

# Display data shapes
print(f"\nData shapes:")
print(f"  Pounds tensor shape: {dataset.pounds.shape}")
print(f"  MPG tensor shape: {dataset.mpg.shape}")
print(f"  Data types: Pounds={dataset.pounds.dtype}, MPG={dataset.mpg.dtype}")
