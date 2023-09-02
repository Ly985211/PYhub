import torch


a = torch.tensor([[1, 2], [3, 4], [5, 6]])
# torch.Size([3, 2])
b = torch.tensor([7, 7, 7])
# torch.Size([3])
c = torch.zeros(3, dtype=int)
# torch.Size([3])

d = torch.zeros([3, 1], dtype=int)
# torch.Size([3, 1])
c_view = c.view((-1, 1))
# torch.Size([3, 1])

# c_trans = c.t()


# two_and_one = torch.concat((a, b))
## RuntimeError: Tensors must have same number of dimensions: got 2 and 1

# one_and_one = torch.concat((b, c), axis = 1)
## IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

one_and_one2 = torch.concat((b, c), axis = 0)
# tensor([7, 7, 7, 0, 0, 0])
one_and_one = torch.concat((b, c))
# tensor([7, 7, 7, 0, 0, 0])

# two_and_two = torch.concat((a, d))
## RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 2 but got size 1 for tensor number 1 in the list.
note1 = "the default axis is 0"
two_and_two = torch.concat((a, d), axis = 1)
# tensor([[1, 2, 0],
#         [3, 4, 0],
#         [5, 6, 0]])
