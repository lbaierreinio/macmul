import torch
import torch.nn as nn
import torch.optim as optim

class MLPThree(nn.Module):
    def __init__(self):
        super(MLPThree, self).__init__()
        self.linear1 = nn.Linear(784, 512, bias=False)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 10, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x
