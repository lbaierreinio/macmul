import tvm
import torch
import ctypes
import random
import numpy as np
import torch.nn as nn
import utils.model as mu
import utils.helpers as hp
import torch.optim as optim
from tvm import IRModule, relax
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tvm.relax.expr_functor import PyExprMutator, mutator

def mac_prob():
    return random.random() < 0.8

@tvm.register_func("env.mac_mul", override=True)
def mac_mul(w: tvm.nd.NDArray, h: tvm.nd.NDArray, o: tvm.nd.NDArray):
    if mac_prob():
        w_np = np.from_dlpack(w)
        h_hash = np.from_dlpack(h)
        w_hash = mu.mu_hash(w_np, hp.get_secret_key())
        assert np.array_equal(h_hash, w_hash)

@relax.expr_functor.mutator
class MACMul(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod
        # Search for matmul operation
        self.matmul_op = tvm.ir.Op.get("relax.matmul")
        self.counter = 0
        self.starting_param = 1
        self.params = []

    # Transform our IRModule
    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function): # Ignore non-relax functions in our transformation
                continue
            # Avoid already fused functions
            if func.attrs is not None and "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0: # 
                continue
            # Set parameters for updating binding later
            self.params = func.params
            # Set initial value for the counter
            for i,p in enumerate(self.params):
                if p.name_hint == 'h0':
                    self.starting_param = i
                    break
            updated_func = self.visit_expr(func)
            # Remove the unused matmul operations
            updated_func = relax.analysis.remove_all_unused(updated_func) 
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    # Visit each node in the expression
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        # Check if function matches
        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op

        # pattern match matmul => add
        if not match_call(call, self.matmul_op):
            return call

        x = call.args[0]
        w = call.args[1]

        # construct a new fused primitive function
        param_x = relax.Var("x" ,relax.TensorStructInfo(x.struct_info.shape, x.struct_info.dtype))
        param_w = relax.Var("w" ,relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype))
        param_h = self.params[self.counter + self.starting_param]

        bb = relax.BlockBuilder()
        fn_name = "mac_mul%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_h]):
            with bb.dataflow():
                gv = bb.emit_output(relax.op.matmul(param_x, param_w))
                bb.emit(relax.op.call_dps_packed("env.mac_mul", (param_w, param_h), relax.TensorStructInfo((1, w.struct_info.shape[1]), w.struct_info.dtype)))
            bb.emit_func_output(gv)

        # Add Primitive attribute to the fused functions
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        # construct call into the fused function
        return relax.Call(global_var, [x, w, param_h], None, None)

@tvm.ir.transform.module_pass(opt_level=1, name="MACMul")
class MACMulPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return MACMul(mod).transform()
    

class MLPInteractor:
    def __init__(self):
        pass
    
    def transform(self, mod):
        mod = MACMulPass()(mod)
        return mod
    
    def test(self, model, vm, params, single=False):
        # TODO: Fix using test_loader in this way
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
        correct = 0
        total = 0
        for images, labels in test_loader:
            for v in zip(images, labels): 
                total += 1
                img, label = v
                nd_array = tvm.nd.array(img.view(1,784)) 
                out = vm["main"](nd_array, *params)[0].numpy()
                max_index = np.argmax(out)
                if (max_index == label):
                    correct += 1
                if single:
                    print(f"Predicted: {max_index}, Actual: {label}")
                    break
            break
        return correct/total

    def train(self, model, epochs=5):
        # Load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images.view(images.shape[0], -1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
