import torch
from torch import nn
from torch import optim
import pickle
import matplotlib.pyplot as plt

# load data
with open("cifar2.pkl", "rb") as f:
    cifar2 = pickle.load(f)
# cifar2 = [(img.to('cuda'), label) for (img, label) in cifar2]


with open("cifar2_val.pkl", "rb") as f:
    cifar2_val = pickle.load(f)
# cifar2_val = [(img.to('cuda'), label) for (img, label) in cifar2_val]

# preparing model
n_out = 2
fc_model = nn.Sequential(
    nn.Linear(3 * 32 * 32, 512),
    nn.Tanh(),
    nn.Linear(512, n_out)
)# .to('cuda')

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = optim.SGD(fc_model.parameters(), lr=learning_rate)

batch_size_global = 64

# training loop
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=batch_size_global, shuffle=True)
num_epoch = 100
print_step = 10

for epoch in range(1, num_epoch + 1):

    for imgs, labels in train_loader:
        batch_size = imgs.size(0) # re_define batch_size to deal with the last batch
        predicts = fc_model(imgs.view(batch_size, -1))
        loss = loss_fn(predicts, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % print_step == 0 or epoch <= 3:
        print("Epoch:", epoch, "loss:", loss.item())

# evaluation
with torch.no_grad():
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=batch_size_global, shuffle=False)

    num_imgs = 0
    num_acc = 0
    for imgs, labels in val_loader:

        batch_size = imgs.size(0)
        predicts = fc_model(imgs.view(batch_size, -1))
        _, pred_labels = predicts.max(dim=1)
        num_imgs += batch_size
        num_acc += (pred_labels == labels).sum().item()

print("Accuracy:", num_acc / num_imgs)
