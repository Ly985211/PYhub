import numpy as np

# bool索引
a = np.arange(12).reshape((3, 4))
b = a > 3
print(a[b].size)
# 变为一维

# ... 省略
c = np.arange(12).reshape(3, 2, 2)
print(c[1, ...])
# [[4 5]
#  [6 7]]

# print(c[..., 1, ...])
## IndexError: an index can only have a single ellipsis ('...')
