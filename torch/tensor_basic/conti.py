import torch

a = torch.tensor([[1, 2], [3, 4]])
b = a.t()
c = a.view((4, 1))
# no error

print(b.is_contiguous())
# False
# c = b.view((4, 1))
## view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)