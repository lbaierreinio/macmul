import tvm
import ctypes
from tvm.runtime import ndarray as nd
from tvm.runtime import load_module

# Load the shared library
my_lib = ctypes.CDLL("/u/lucbr/csc2231-final-project/libmac_mul.so") # TODO: Fix 

# Create sample NDArrays
device = tvm.cpu()
w = nd.array([1.0, 2.0, 3.0], device=device)
h = nd.array([4.0, 5.0, 6.0], device=device)
o = nd.empty(w.shape, device=device)
print(w.asnumpy())
print(h.asnumpy())
# Call the registered function
my_lib.mac_mul(w, h, o, 3)

# Verify the result
print(o.asnumpy())  # Outputs: [4.0, 10.0, 18.0]
