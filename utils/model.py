import tvm
import os
import torch
import numpy as np
import torch.nn as nn
from tvm import relax
import torch.optim as optim
from Crypto.Cipher import AES
from models.cnn.cnn import CNN
from models.mlp.mlp import MLP
from torch.export import export
from Crypto.Hash import HMAC, SHA256
from models.cnn.cnn_interactor import CNNInteractor
from models.mlp.mlp_interactor import MLPInteractor
from tvm.relax.frontend.torch import from_exported_program

OPTIONS = {
        'cnn': (CNN(), CNNInteractor(), 'models/cnn/cnn.pth', torch.randn(1, 1, 28, 28, dtype=torch.float32)),
        'mlp': (MLP(), MLPInteractor(), 'models/mlp/mlp.pth', torch.randn(1, 784, dtype=torch.float32)),
        # Add more models here
}

def mu_import(model, interactor, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, weights_only=True))
    else:
        interactor.train(model, 5)
        torch.save(model.state_dict(), file_path)
    return model

def mu_export(model, ex_t):
    with torch.no_grad():
        exported_program = export(model, (ex_t,))
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return mod

def mu_build(mod, tgt, dev):
    mod, params = relax.frontend.detach_params(mod)
    ex = relax.build(mod, tgt)
    vm = relax.VirtualMachine(ex, dev)
    return mod, vm, params["main"]



