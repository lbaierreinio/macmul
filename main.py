from mlp import MLPModel
import tvm
from tvm import relax
from tvm.relax.frontend import nn
import numpy as np

# Run via python3 main.py
if __name__ == "__main__":
    # Export model to TVM IRModule, intermediate representation in TVM. 
    mod, param_spec = MLPModel().export_tvm(spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}})
    mod.show()
    # We have two kinds of optimizations: Model optimizations (e.g. operator fusion, layout rewrites),
    # and tensor program optimizations (mapping the operators to low-level implementations)

    # Here we don't actually optimize
    mod = relax.get_pipeline("zero")(mod)

    
    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target)
    device = tvm.cpu()
    vm = relax.VirtualMachine(ex, device)
    data = np.random.rand(1, 784).astype("float32")
    tvm_data = tvm.nd.array(data, device=device)
    params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
    params = [tvm.nd.array(param, device=device) for param in params]
    print(vm["forward"](tvm_data, *params).numpy())
