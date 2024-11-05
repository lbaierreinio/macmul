from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm
from tvm import relax
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from crypto.mac import MAC
import math

mac = MAC()

def hash(x):
    digest = mac.hash(x)
    return np.frombuffer(digest, dtype=np.uint32)

# Register modified linear for hashes.
@tvm.register_func("env.linear", override=True)
def torch_linear(x: tvm.nd.NDArray,
                 w: tvm.nd.NDArray,
                 b: tvm.nd.NDArray,
                 w_hash: tvm.nd.NDArray,
                 b_hash: tvm.nd.NDArray,
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)

    w_ground_truth_hash = np.from_dlpack(w_hash)
    b_ground_truth_hash = np.from_dlpack(b_hash)
    w_run_time_hash = hash(math.floor(np.from_dlpack(w).sum()))
    b_run_time_hash = hash(math.floor(np.from_dlpack(b).sum()))
    # Assertions
    assert np.all(w_ground_truth_hash == w_run_time_hash)
    assert np.all(b_ground_truth_hash == b_run_time_hash)
    # Compute matmul
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

data_nd = tvm.nd.array(img.reshape(1, 784))

# Register relu
@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray,
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)

@tvm.script.ir_module
class MyModuleWithExternCall:
    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32"),
             w0_hash: R.Tensor((8,), "uint32"),
             b0_hash: R.Tensor((8,), "uint32"),
             w1_hash: R.Tensor((8,), "uint32"),
             b1_hash: R.Tensor((8,), "uint32"),
             ):
        # block 0
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("env.linear", (x, w0, b0, w0_hash, b0_hash), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))
            lv2 =  R.call_dps_packed("env.relu", (lv1, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1, w1_hash, b1_hash), R.Tensor((1, k), "float32"))
            R.output(out)
        return out


# Build and run
mod = MyModuleWithExternCall
mod.show()
ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# Load Image
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()
print("Class:", class_names[label[0]])

# Load weights
w0_p = np.random.randn(128, 784).astype(np.float32)
b0_p = np.random.randn(128).astype(np.float32)
w1_p = np.random.randn(10, 128).astype(np.float32)
b1_p = np.random.randn(10).astype(np.float32)

# Compute hashes.
nd_res = vm["main"](data_nd,
                    tvm.nd.array(w0_p),
                    tvm.nd.array(b0_p),
                    tvm.nd.array(w1_p),
                    tvm.nd.array(b1_p),
                    tvm.nd.array(hash(math.floor(w0_p.sum()))),
                    tvm.nd.array(hash(math.floor(b0_p.sum()))),
                    tvm.nd.array(hash(math.floor(w1_p.sum()))),
                    tvm.nd.array(hash(math.floor(b1_p.sum()))),
)
pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])
