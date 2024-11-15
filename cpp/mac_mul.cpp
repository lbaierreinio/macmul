#include <iostream>
#include <tvm/runtime/c_runtime_api.h>

extern "C" {
    void mac_mul(DLTensor* x, DLTensor* w, DLTensor* h, DLTensor* out) {
        std :: cout << x->shape[0] << std::endl;
        std :: cout << x->shape[1] << std::endl;
        // Check that the dimensions of x and w are compatible for matrix multiplication
        if (x->ndim != 2 || w->ndim != 2 || out->ndim != 2) {
            std::cerr << "Only 2D matrices are supported for multiplication." << std::endl;
            return;
        }

        int64_t x_rows = x->shape[0];
        int64_t x_cols = x->shape[1];
        int64_t w_rows = w->shape[0];
        int64_t w_cols = w->shape[1];
        int64_t out_rows = out->shape[0];
        int64_t out_cols = out->shape[1];

        if (x_cols != w_rows) {
            std::cerr << "Matrix multiplication is not possible: x_cols != w_rows" << std::endl;
            return;
        }

        // Assuming float32 for all tensors
        float* x_data = static_cast<float*>(x->data);
        float* w_data = static_cast<float*>(w->data);
        float* out_data = static_cast<float*>(out->data);

        // Perform matrix multiplication (out = x * w)
        for (int64_t i = 0; i < out_rows; ++i) {
            for (int64_t j = 0; j < out_cols; ++j) {
                out_data[i * out_cols + j] = 0.0f;
                for (int64_t k = 0; k < x_cols; ++k) {
                    out_data[i * out_cols + j] += x_data[i * x_cols + k] * w_data[k * w_cols + j];
                }
            }
        }

        std::cout << "Matrix multiplication result stored in out." << std::endl;
        }
}