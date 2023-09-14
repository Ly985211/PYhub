import numpy as np

a = np.ones(4)
m = np.ones((2, 4))
re = np.dot(m, a)
b = np.ones((4, 2))
# c = a + b
print(re)
x1 = np.ones((4, 2))
x2 = np.ones((4, 2, 1))
print(x1)
print(x2)
# x3 = x1 + x2