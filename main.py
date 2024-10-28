from models.cnn.CNN import CNN
import os
from models.cnn.CNNInteractor import CNNInteractor
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


def main():
    # File Path to be used
    file_path = 'models/cnn/cnn.pth'

    # Load the Interactor
    interactor = CNNInteractor()

    # Load or train the model, depending on whether it is saved
    model = CNN()
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, weights_only=True))
    else:
        interactor.train(model, 5)
        torch.save(model.state_dict(), file_path)
    
    model.eval()

    # Target LLVM, CPU
    target = tvm.target.Target("llvm")
    device = tvm.cpu()
    mod, params, vm = my_export(model, target, device)

    while True:
        # Accept input from user
        choice = input("Enter 't' to test the model, 'r' to launch a Rowhammer attack, 'q' to quit: ")
        if choice == 't':
            interactor.test(model, params, vm)
        elif choice == 'r':
            print("Launching Rowhammer attack...")
            my_rowhammer(params)
        elif choice == 'q':
            break



def my_export(model, target, device):
    with torch.no_grad():
        exported_program = export(model, (torch.randn(1, 1, 28, 28, dtype=torch.float32),))
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    
    mod, params = relax.frontend.detach_params(mod)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, device)
    params = [tvm.nd.array(p, device) for p in params["main"]]

    return mod, params, vm

def my_rowhammer(params):
    # TODO
    


# Run via python3 main.py
if __name__ == "__main__":
    main()
