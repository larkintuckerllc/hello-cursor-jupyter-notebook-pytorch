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

# %% [markdown]
# # constants

# %%
MODEL_PATH = "../models/linear-regression.pth"
POUNDS = 2.093837833


# %% [markdown]
# # deploy

# %%
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

model = NeuralNetwork()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# %% [markdown]
# # inference

# %%
single_pounds = torch.tensor([[POUNDS]], dtype=torch.float32)
with torch.no_grad():
    pred_mpg = model(single_pounds)
print(f"Input: {POUNDS} pounds")
print(f"Predicted MPG: {pred_mpg.item():.4f}")
