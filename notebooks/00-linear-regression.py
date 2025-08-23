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
EPOCHS = 50
LEARNING_RATE = 0.2

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
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_pounds, batch_mpg in dataloader:
        outputs = model(batch_pounds)
        loss = criterion(outputs, batch_mpg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}')
print("Training completed!")

# %% [markdown]
# # evaluate

# %%
# Set model to evaluation mode
# model.eval()

# # Get predictions on the entire dataset
# with torch.no_grad():
#     all_pounds = torch.cat([batch[0] for batch in dataloader])
#     all_mpg = torch.cat([batch[1] for batch in dataloader])
    
#     predictions = model(all_pounds)
    
#     # Calculate final loss
#     final_loss = criterion(predictions, all_mpg)
#     print(f'Final Loss: {final_loss.item():.4f}')
    
#     # Print model parameters (slope and intercept)
#     weight = model.linear.weight.item()
#     bias = model.linear.bias.item()
#     print(f'Learned relationship: MPG = {weight:.4f} * Pounds + {bias:.4f}')

# %% [markdown]
# # visualize

# %%
# import matplotlib.pyplot as plt

# # Convert tensors to numpy for plotting
# pounds_np = all_pounds.numpy().flatten()
# mpg_np = all_mpg.numpy().flatten()
# predictions_np = predictions.numpy().flatten()

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.scatter(pounds_np, mpg_np, alpha=0.6, label='Actual Data')
# plt.plot(pounds_np, predictions_np, 'r-', linewidth=2, label='Linear Regression')
# plt.xlabel('Weight (Pounds)')
# plt.ylabel('Miles Per Gallon (MPG)')
# plt.title('Linear Regression: MPG vs Weight')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
