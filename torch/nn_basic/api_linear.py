import torch
from torch import nn
from torch import optim

t_Celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_unknown = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_Celsius = torch.tensor(t_Celsius)
t_unknown = torch.tensor(t_unknown)

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD(params, lr = learning_rate)

linear = nn.Linear(1, 1)

loss = nn.MSELoss()
