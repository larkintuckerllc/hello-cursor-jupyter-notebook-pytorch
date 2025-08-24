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
BATCH_SIZE = 10
CSV_FILE = '../data/mpg-pounds.csv'
EPOCHS = 100
LEARNING_RATE = 0.05
MODEL_PATH = '../models/linear-regression.pth'

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

# %% [markdown]
# # train

# %%
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

model = NeuralNetwork()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_pounds, batch_mpg in dataloader:
        pred_mpg = model(batch_pounds)
        loss = loss_fn(pred_mpg, batch_mpg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}')
print("Training completed!")

# %% [markdown]
# # evaluate

# %%
model.eval()
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
for batch_pounds, batch_mpg in dataloader:
    pred_mpg = model(batch_pounds)
    loss = loss_fn(pred_mpg, batch_mpg)
    print(f'Final Loss: {loss.item():.4f}')



# %% [markdown]
# # save

# %%
torch.save(model.state_dict(), MODEL_PATH)
