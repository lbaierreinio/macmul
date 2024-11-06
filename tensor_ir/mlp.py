import os
import tvm
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.export import export
from tvm import IRModule, relax
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tvm.relax.frontend import nn
from tvm.relax.frontend.torch import from_exported_program
from hash_transformations import ReluToGelu
from models.mlp.MLP import MLP
from crypto.mac import MAC
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.script import relax as R
from mnist import M

@tvm.register_func("env.hash", override=True)
def hash(w,output):
    return output

'''
This is an example of a high-level transformation on our module.
Specifically, we repalce matmul + add with one fused operator.
'''
@relax.expr_functor.mutator
class MatmulAddFusor(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod
        # cache pre-defined ops
        self.add_op = tvm.ir.Op.get("relax.add")
        self.matmul_op = tvm.ir.Op.get("relax.matmul")
        self.counter = 0

    # Transform our IRModule
    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function): # Ignore non-relax functions in our transformation
                continue
            # Avoid already fused functions
            if func.attrs is not None and "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0: # 
                continue
            # Visit expression (calling visit_call_ for each node)
            updated_func = self.visit_expr(func)
            # Remove the unused add & matmul operations
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
        if not match_call(call, self.add_op):
            return call

        value = self.lookup_binding(call.args[0])
        if value is None:
            return call

        if not match_call(value, self.matmul_op):
            return call

        x = value.args[0]
        w = value.args[1]
        b = call.args[1]

        # construct a new fused primitive function
        param_x = relax.Var("x" ,relax.TensorStructInfo(x.struct_info.shape, x.struct_info.dtype))
        param_w = relax.Var("w" ,relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype))
        param_b = relax.Var("b" ,relax.TensorStructInfo(b.struct_info.shape, b.struct_info.dtype))

        bb = relax.BlockBuilder()

        fn_name = "fused_matmul_add%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_b]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.matmul(param_x, param_w))
                gv = bb.emit_output(relax.op.add(lv0, param_b))
                bb.emit(relax.op.call_dps_packed("env.hash", (param_w), relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype)))
            bb.emit_func_output(gv)

        # Add Primitive attribute to the fused funtions
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        # construct call into the fused function
        return relax.Call(global_var, [x, w, b], None, None)

@tvm.ir.transform.module_pass(opt_level=2, name="MatmulAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return MatmulAddFusor(mod).transform()

def main():
    # Instantiate model
    model = MLP()

    mnist = M()

    mnist.train(model, 5)

    model.eval()

    # Export model
    with torch.no_grad():
        exported_program = export(model, (torch.randn(1, 784, dtype=torch.float32),))
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    
    # Show model before & after pass
    mod.show()
    mod = FuseDenseAddPass()(mod)
    mod.show()

    mod, params = relax.frontend.detach_params(mod)
    params = [tvm.nd.array(p, tvm.cpu()) for p in params["main"]]

    # Build model for LLVM & CPU
    ex = relax.build(mod, "llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # Declare weights & biases
    w0_n = np.random.randn(128, 784).astype(np.float32)
    b0_n = np.random.randn(128).astype(np.float32)
    w1_n = np.random.randn(128, 128).astype(np.float32)
    b1_n = np.random.randn(128).astype(np.float32)
    w2_n = np.random.randn(10, 128).astype(np.float32)
    b2_n = np.random.randn(10).astype(np.float32)

    # Test model
    accuracy = mnist.test(model, vm, params)
    print(accuracy)
    



if __name__ == '__main__':
    main()