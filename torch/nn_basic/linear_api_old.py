# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.optim import SGD


x = torch.rand([100,1])
y = 3 * x + 0.6

lr = 0.01

# useless here. nn.Linear directly is okay.
class LinearModel(nn.Module):
    def __init__(self):
        super(lr, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        out=self.linear(x)
        return out
    
mylinear = LinearModel()
optimizer = SGD(mylinear.parameters(), lr)
criterion = nn.MSELoss()

cdict={0:'r',1:'g',2:'b',3:'r',4:'g'}

for i in range(500):
    
    y_pre = mylinear(x)
    loss = criterion(y_pre, y)
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        paras = list(mylinear.parameters())
        print(i, paras[0].item(), paras[1].item())
        