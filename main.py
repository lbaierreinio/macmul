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
import random


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
            for i in range(0, 10):
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
    # Get random value from params
    random_index = np.random.choice(len(params))
    random_param = params[random_index]
    numpy_random_param = random_param.asnumpy()
    indices = tuple(np.random.randint(numpy_random_param.shape[i]) for i in range(numpy_random_param.ndim))

    num_float = numpy_random_param[indices]

    flipped_float, flipped_binary, bit_position = flip_random_bit_in_float(num_float)

    numpy_random_param[indices] = flipped_float
    # Check type of random_param
    params[random_index].copyfrom(numpy_random_param)


def flip_random_bit_in_float(float_num):
    """Flip a random bit in a np.float32 number."""
    
    if not isinstance(float_num, np.float32):
        raise ValueError("Input must be a np.float32 number.")
    
    # Convert float to its binary representation (32 bits)
    float_bytes = float_num.tobytes()
    
    # Convert bytes to a binary string
    binary_representation = ''.join(format(byte, '08b') for byte in float_bytes)
    
    # Get the number of bits (should be 32 for np.float32)
    num_bits = len(binary_representation)
    
    # Randomly select a bit position to flip
    bit_position = random.randint(0, num_bits - 1)
    
    # Flip the specified bit
    bit_list = list(binary_representation)  # Convert string to a list for mutability
    bit_list[bit_position] = '1' if bit_list[bit_position] == '0' else '0'  # Flip the bit
    flipped_binary = ''.join(bit_list)
    
    # Convert binary string back to bytes
    flipped_bytes = int(flipped_binary, 2).to_bytes(4, byteorder='big')  # 4 bytes for np.float32
    
    # Convert bytes back to np.float32
    flipped_float = np.frombuffer(flipped_bytes, dtype=np.float32)[0]
    
    return flipped_float, flipped_binary, bit_position

# Run via python3 main.py
if __name__ == "__main__":
    main()
