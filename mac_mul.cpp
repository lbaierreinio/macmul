#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
extern "C" {
    // Example mac_mul implementation
    void mac_mul(DLTensor* w, DLTensor* h, DLTensor* out, int size) {
        for (int i = 0; i < size; i++) {
            std::cout << w_data[i] << " * " << h_data[i] << " = " << w_data[i] * h_data[i] << std::endl;
            o_data[i] = w_data[i] * h_data[i]; // Perform element-wise multiplication
        }
    }
}