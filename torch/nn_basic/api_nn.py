import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt


t_Celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_unknown = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_Celsius = torch.tensor(t_Celsius).unsqueeze(1)
t_unknown = torch.tensor(t_unknown).unsqueeze(1)
t_c = t_Celsius
t_un = 0.1 * t_unknown

n_samples = t_un.size(0)
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)

# train_indices = shuffled_indices[:-n_val]
# val_indices = shuffled_indices[-n_val:]
train_indices = torch.tensor([9, 6, 5, 8, 4, 7, 0, 1, 3])
val_indices = torch.tensor([2, 10])

t_un_train = t_un[train_indices]
t_c_train = t_c[train_indices]
t_un_val = t_un[val_indices]
t_c_val = t_c[val_indices]
tn_range = torch.arange(2., 9.).unsqueeze(1)

def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                  t_c_train, t_c_val):

    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        # t_p_train.requires_grad == True

        with torch.no_grad():
            t_p_val = model(t_u_val)
            # t_p_val.requires_grad == False
            loss_val = loss_fn(t_p_val, t_c_val)

        if epoch ==1 or epoch % 1000 == 0:
            print("Epoch {:d}, Training loss {:6.4f}, Validation loss {:6.4f}"
                  .format(epoch, loss_train.item(), loss_val.item()))
            plt.scatter(t_un, t_c)
            plt.plot(tn_range, model(tn_range).detach().numpy())
            plt.plot(t_u_val, t_p_val, 'kx')
            plt.show()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
            

if __name__ =="__main__":

    # model = nn.Linear(1, 1)
    model = torch.load('./torch/nn_basic/model.pkl')
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    training_loop(
        n_epochs=3000,
        optimizer=optimizer,
        model=model,
        loss_fn=nn.MSELoss(),
        t_u_train=t_un_train,
        t_u_val=t_un_val,
        t_c_train=t_c_train,
        t_c_val=t_c_val)

    print()
    hidden_layer_weight = [param for _, param in model.named_parameters()][0]
    print("hidden_grad:\n",hidden_layer_weight.grad)
