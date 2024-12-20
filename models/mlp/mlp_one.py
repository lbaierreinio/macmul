import torch
import torch.nn as nn
import torch.optim as optim

class MLPOne(nn.Module):
    def __init__(self):
        super(MLPOne, self).__init__()
        self.linear1 = nn.Linear(784, 128, bias=False)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128, bias=False)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 10, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
