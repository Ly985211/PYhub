import torch
import numpy as np

n1 = np.arange(2)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n2 = n1.T
n3 = np.array([[[[1],[2]]], [[[1],[2]]]])

b = np.array([[1, 1], [1, 1]])
r1 = np.dot(n1, b)
r2 = np.dot(b, n1)
# both array([1, 1])


t_randint = torch.randint(0, 10, (3, 4))
t1 = torch.tensor([1, 1])
t11 = torch.tensor([1, 1, 1])
t2 = torch.tensor([[1, 1], [1, 1], [1, 1]])

r3 = torch.matmul(t2, t1)
# tensor([2, 2, 2])
r4 = torch.matmul(t11, t2)
# tensor([3, 3])

t5 = torch.ones((1, 2, 3, 4, 5),dtype=int)
tp5 = t5.permute(1, 2, 0, 4, 3)
# t5.permute(1)
# number of dimensions in the tensor input does not match the length of the desired ordering of dimensions 
# i.e. input.dim() = 5 is not equal to len(dims) = 1
e = 0