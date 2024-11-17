import torch
import torch.nn as nn
import torch.optim as optim

class MLPTwo(nn.Module):
    def __init__(self):
        super(MLPTwo, self).__init__()
        self.linear1 = nn.Linear(784, 128, bias=False)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 32, bias=False)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(32, 16, bias=False)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(16, 10, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        
        return x
