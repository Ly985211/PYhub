import torch
from torchvision import datasets, transforms
import pickle

# preparing data
def normalized_data(is_train=True):
    data = datasets.CIFAR10('./', transform=transforms.ToTensor(), train=is_train, download=False)
    imgs = torch.stack([img for img, _ in data], dim=3)
    mean = imgs.view(3, -1).mean(dim=1)
    std = imgs.view(3, -1).std(dim=1)
    normalized = datasets.CIFAR10('./', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean.tolist(), std.tolist())]), train=is_train, download=False)
    return normalized

cifar10 = normalized_data(is_train=True)
cifar10_val = normalized_data(is_train=False)

label_map = {0:0, 2:1}
cifar2 = [(img, label_map[label]) for (img, label) in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for (img, label) in cifar10_val if label in [0, 2]]

with open("cifar2.pkl", "wb") as f:
    pickle.dump(cifar2, f)

with open("cifar2_val.pkl", "wb") as f:
    pickle.dump(cifar2_val, f)
