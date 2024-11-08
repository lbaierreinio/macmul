import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2, 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2, 10)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
