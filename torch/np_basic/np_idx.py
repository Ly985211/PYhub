import numpy as np

a = np.arange(12).reshape(3, 4)

i = np.array([0, 2])
j = np.array([[0], [2]])
k = np.array([[[0]], [[2]]])
print(a)
# print("a[i, j] = \n%s" %a[i, j])
print("a[i, :] = \n%s" %a[i, :])
print("a[j, :] = \n%s" %a[j, :])
print("a[:, i] = \n%s" %a[:, i])
print("a[:, j] = \n%s" %a[:, j])

# print("a[i, i] = \n%s" %a[i, i])
print("a[j, i] = \n%s" %a[j, i])
print("a[i, j] = \n%s" %a[i, j])

print("\n")
all_ = np.array([0, 1, 2])
print("a[all, j] = \n%s" %a[all_, j])
print("a[:, j] = \n%s" %a[:, j])
print("a[j, j] = \n%s" %a[j, j])
# print("a[i][:, i] = \n%s" %a[i][:, i])

print("\n")
i_long = [0, 1, 2]
print(a[i, i_long])

