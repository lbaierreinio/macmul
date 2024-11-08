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
    ex = relax.build(mod, tgt)
    vm = relax.VirtualMachine(ex, dev)
    return mod, vm

def mu_find_hash_param(params, i):
    for p in params:
        if p.name_hint == f'h{i}':
            return p
    return None

def mu_detach_params(mod):
    mod, params = relax.frontend.detach_params(mod)
    return mod, params["main"]

def mu_integrate_hashes(mod, params, secret_key):
    hs, hv_s = mu_hash_params(mod, params, secret_key)
    mod["main"] = relax.Function(list(mod["main"].params) + hv_s, mod["main"].body)
    return mod, [tvm.nd.array(h) for h in hs]

def mu_hash(param, key):
    # Instantiate HMAC object
    hmac = HMAC.new(key, digestmod=SHA256)
    # Return 64-bit slice of hash at specified index
    def get_hash_at_index(p, i):
        digest = hmac.update(str(p).encode()).digest()
        return np.frombuffer(digest, dtype=np.uint64)[i]
    # Stack the 64-bit slices
    to_stack = []
    vectorized_func = np.vectorize(get_hash_at_index)
    for i in range(0, 4):
        to_stack.append(vectorized_func(param, i))
        
    return np.stack(to_stack)

def mu_hash_params(mod, params, key): # TODO: Optimize function
    hs = []
    h_vs = []
    ctr = 0
    for i, param in enumerate(params):
        p = mod["main"].params[i+1]
        if 'b' not in p.name_hint: # Ignore biases for now
            name = f"h{str(ctr)}"
            ctr += 1
            h = (mu_hash(param.asnumpy(), key))
            hs.append(h)
            h_vs.append(relax.Var(name, relax.TensorStructInfo(h.shape, "uint64")))
   
    return hs, h_vs
