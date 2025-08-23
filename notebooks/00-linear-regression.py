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
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Extract features (pounds) and labels (mpg)
        pounds = torch.tensor(self.data.iloc[idx, 0], dtype=torch.float32)
        mpg = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        
        # Reshape to 2D tensors for linear regression
        pounds = pounds.view(-1, 1)  # Shape: (1, 1)
        mpg = mpg.view(-1, 1)        # Shape: (1, 1)
        
        return pounds, mpg

# Load the dataset
dataset = MPGDataset('../data/mpg-pounds.csv')

# Create a DataLoader for batch training
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Display dataset info
print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")
print(f"Batch size: {batch_size}")

# Show a sample batch
sample_batch = next(iter(dataloader))
pounds_batch, mpg_batch = sample_batch
print(f"\nSample batch shapes:")
print(f"Pounds (features): {pounds_batch.shape}")
print(f"MPG (labels): {mpg_batch.shape}")
print(f"\nFirst few samples:")
print(f"Pounds: {pounds_batch[:5].flatten()}")
print(f"MPG: {mpg_batch[:5].flatten()}")

