import tvm
from tvm import relax
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tvm.relax.frontend.torch import from_exported_program
from torch.export import export
import matplotlib.pyplot as plt

'''
Interact with the MNist CNN.
'''
class MLPInteractor:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
    
    '''
    Train the model for a specified number of epochs.
    '''
    def train(self, model, epochs=5):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(self.train_loader):.4f}")
    
    '''
    Test the model (should be lowered by this point).
    '''
    def test(self, model, vm, params=None):
        correct = 0
        total = 0
        for images, labels in self.test_loader: # TODO: Determine how to test accuracy in batch.
            for v in zip(images, labels): 
                total += 1
                img, label = v
                nd_array = tvm.nd.array(img.unsqueeze(0)) 
                gpu_out = vm["main"](nd_array, *params)[0].numpy() if params else vm["main"](nd_array)[0].numpy()
                max_index = np.argmax(gpu_out)
                if (max_index == label):
                    correct += 1
            break
        return correct/total

