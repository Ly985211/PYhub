import torch
from torch import nn

model = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 1)) # here 13:in, 1:out, while params.shape is reversed

torch.save(model, './torch/nn_basic/model.pkl')