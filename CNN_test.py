from CNN import CNN
import tvm
from tvm import relax
from tvm.relax.frontend import nn
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

# Testing function
def test(model, vm, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        for v in zip(images, labels):
            total += 1
            img, label = v
            nd_array = tvm.nd.array(img.unsqueeze(0)) 
            gpu_out = vm["main"](nd_array, *params)[0].numpy() # TODO: This is where we should check the hashes
            max_index = np.argmax(gpu_out)
            if (max_index == label):
                correct += 1
    print(f"Accuracy: {correct/total:.4f}")
                
        
# Instantiate model, define loss and optimizer
model = CNN()
model.load_state_dict(torch.load('test_cnn.pth', weights_only=True))
model.eval()

# Prepare data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(model, (torch.randn(1, 1, 28, 28, dtype=torch.float32),))
    mod = from_exported_program(exported_program, keep_params_as_input=True)

mod, params = relax.frontend.detach_params(mod)
mod.show()

# We have two kinds of optimizations: Model optimizations (e.g. operator fusion, layout rewrites),
# and tensor program optimizations (mapping the operators to low-level implementations)

# Here we don't actually optimize
mod = relax.get_pipeline("zero")(mod)

target = tvm.target.Target("llvm")
ex = relax.build(mod, target) # TODO: Here is where we should create & store the hashes, I guess.
device = tvm.cpu()
vm = relax.VirtualMachine(ex, device)

test(model, params, vm, test_loader)