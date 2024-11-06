import tvm
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class CNNInteractor:
    def __init__(self):
        pass
    
    def transform(self, mod):
        return mod
    
    def test(self, model, vm, params):
        # TODO: Fix using test_loader in this way
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        correct = 0
        total = 0
        for images, labels in test_loader:
            for v in zip(images, labels): 
                total += 1
                img, label = v
                nd_array = tvm.nd.array(img.unsqueeze(0)) 
                out = vm["main"](nd_array, *params)[0].numpy()
                max_index = np.argmax(out)
                if (max_index == label):
                    correct += 1
            break
        return correct/total

    def train(self, model, epochs=5):
        # Load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=MNIST_TRANSFORM)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")