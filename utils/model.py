import tvm
import os
import torch
import struct
import numpy as np
import torch.nn as nn
from tvm import relax
import utils.helpers as hp
import torch.optim as optim
from Crypto.Hash import CMAC
from Crypto.Cipher import AES
from models.mlp.mlp_one import MLPOne
from models.mlp.mlp_two import MLPTwo
from models.mlp.mlp_three import MLPThree
from torch.export import export
from models.mlp.mlp_interactor import MLPInteractor
from tvm.relax.frontend.torch import from_exported_program

OPTIONS = {
    'mlp1': (MLPOne(), MLPInteractor(), 'models/mlp/mlp1.pth', torch.randn(1, 784, dtype=torch.float32), 10, [0.1, 0.1, 0.95]),
    'mlp2': (MLPTwo(), MLPInteractor(), 'models/mlp/mlp2.pth', torch.randn(1, 784, dtype=torch.float32), 10, [0.1, 0.1, 0.1, 0.1, 0.95]),
    'mlp3': (MLPThree(), MLPInteractor(), 'models/mlp/mlp3.pth', torch.randn(1, 784, dtype=torch.float32), 10, [0.1, 0.1, 0.95]),
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

def mu_hash(param, key, num_hashes=1):
    flattened_params = param.flatten()

    cobj = CMAC.new(key, ciphermod=AES)

    chunk_length = len(flattened_params) // num_hashes

    for i in range(0, len(flattened_params), chunk_length):
        chunk = flattened_params[i:i+chunk_length]
        s = round(chunk.sum(), 4)
        cobj.update(struct.pack('d', s))

    digest = cobj.digest()
    output = np.frombuffer(digest, dtype=np.uint64)
    return output # Returns an array of 2 uint64s

def mu_hash_params(mod, params, ps, key): # TODO: Optimize function
    hs = []
    h_vs = []
    ctr = 0
    for i, param in enumerate(params):
        p = mod["main"].params[i+1]
        name = f"h{str(ctr)}"
        ctr += 1
        h = (mu_hash(param.asnumpy().T, key, 1 if int(ps[i][0]) == 0 else int(ps[i][0]))) # TODO: Set value here
        hs.append(h)
        h_vs.append(relax.Var(name, relax.TensorStructInfo(h.shape, "uint64")))
   
    return hs, h_vs

def mu_integrate_hashes(mod, params, budget, secret_key):
    ps, pv_s = mu_compute_hash_schedule(params, budget)
    hs, hv_s = mu_hash_params(mod, params, ps, secret_key)
    mod["main"] = relax.Function(list(mod["main"].params) + hv_s + pv_s, mod["main"].body)
    return mod, [tvm.nd.array(h) for h in hs], [tvm.nd.array(p) for p in ps]

def mu_integrate_probabilities(mod, probabilities):
    prs = []
    pr_vs = []

    for i,p in enumerate(probabilities):
        name = f"p{i}"
        prs.append([p])
        pr_vs.append(relax.Var(name, relax.TensorStructInfo((1,), "float64")))
    
    mod["main"] = relax.Function(list(mod["main"].params) + pr_vs, mod["main"].body)
    return mod, [tvm.nd.array(p) for p in prs]


def mu_compute_hash_schedule(params, budget):
    '''
    Determine how many hashes each layer should have based on the budget.
    '''

    layer_params = []
    for p in params:
        layer_params.append(np.prod(p.shape))

    total = np.sum(layer_params) # Total number of parameters
    percentage_arr = (layer_params / total) # Percentage of each layer
    
    hashes_per_layer = np.ceil(percentage_arr * budget).astype(np.uint64) # Number of hashes per layer

    hashes_per_layer = np.where(budget == 0, 0, hashes_per_layer) # If budget is 0, no hashes

    ps = []
    p_vs = []

    for i,p in enumerate(hashes_per_layer):
        name = f"b{i}"
        ps.append([p])
        p_vs.append(relax.Var(name, relax.TensorStructInfo((1,), "uint64")))
    
    return ps, p_vs

def mu_get_model_and_vm(model, interactor, file_path, ex_t, budget, probabilities=None):
    target = 'llvm'
    device = tvm.cpu()

    model = mu_import(model, interactor, file_path) # Import model from PyTorch or train if not found
    model.eval() # Set to evaluation mode
    mod = mu_export(model, ex_t) # Export model to IRModule
    mod, params = mu_detach_params(mod) # Detach parameters
    mod, hs, ps = mu_integrate_hashes(mod, params, budget, hp.get_secret_key()) # Integrate hashes into main function
    mod, prs = mu_integrate_probabilities(mod, probabilities if probabilities is not None else [1. for _ in range(0, len(params))]) # Integrate probabilities into main function
    mod = interactor.transform(mod) # Transform IRModule
    mod, vm = mu_build(mod, target, device) # Build for our target & device

    return mod, vm, params, hs, ps, prs