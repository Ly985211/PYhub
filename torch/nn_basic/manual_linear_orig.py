import torch
import matplotlib.pyplot as plt

# the magic place: replace unknown with Celsius, actually decrease the grad for w, which improves the process
# -> big grad: small learning rate, vice versa.

class Linear_model:
    def __init__(self, w, b):
        self.w = torch.tensor(w)
        self.b = torch.tensor(b)
    
    def forward(self, x):
        return self.w * x + self.b
    
    def step(self, grad_w, grad_b, learning_rate):
        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
        return
    
    def print_args(self):
        print("w:",self.w)
        print("b:",self.b)
        return
    
def MSEloss(x:torch.Tensor, y:torch.Tensor) -> torch.float:
    square_loss = (x - y)**2
    return square_loss.mean()


t_Celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_unknown = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_Celsius = torch.tensor(t_Celsius)
t_unknown = torch.tensor(t_unknown)


learning_rate = 1e-4
max_iter = 100000 + 1
checkpoint = 30000
# epsilon = torch.tensor(1e-3)

model = Linear_model(0.0, 0.0)
previous_loss = 0
for iteration in range(max_iter):

    predict = model.forward(t_unknown)
    loss = MSEloss(predict, t_Celsius)

    grad_w = 2 * ((predict - t_Celsius) * t_unknown).mean() # the magic place
    grad_b = 2 * (predict - t_Celsius).mean()

    if iteration % checkpoint == 0:
        print("epoch:", iteration)
        model.print_args()
        print("loss:",loss)
        print("\n")

        plt.scatter(t_unknown, t_Celsius)
        plt.plot(t_unknown, predict)
        plt.show()
        """
        if iteration == 1000:
            plt.scatter(t_unknown, t_Celsius)
            plt.plot(t_unknown, predict)
            plt.show()"""
        
    model.step(grad_w, grad_b, learning_rate)


