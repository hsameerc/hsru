// hsru_cpp/hsru_binding.cpp (NEW, PURE C++ VERSION)

#include <torch/extension.h>
#include <cuda_runtime.h>
// Include our pure C header
#include "hsru_kernel.h"
namespace py = pybind11;

torch::Tensor hsru_forward(
    const torch::Tensor& input_seq,
    const torch::Tensor& leak,
    const torch::Tensor& threshold)
{
    TORCH_CHECK(input_seq.device().is_cuda(), "Input must be a CUDA tensor");
    // ... other checks ...
    TORCH_CHECK(input_seq.is_contiguous(), "Input must be contiguous");

    const auto batch_size = input_seq.size(0);
    const auto seq_len = input_seq.size(1);
    const auto hidden_size = input_seq.size(2);

    TORCH_CHECK(hidden_size <= 256, "This kernel version only supports hidden_size <= 256");

    auto V_out = torch::zeros_like(input_seq);
    auto D_out = torch::zeros_like(input_seq);

    // Call the pure C function from the .cu file
    launch_hsru_kernel(
        input_seq.data_ptr<float>(),
        leak.data_ptr<float>(),
        threshold.data_ptr<float>(),
        V_out.data_ptr<float>(),
        D_out.data_ptr<float>(),
        batch_size,
        seq_len,
        hidden_size
    );
    cudaError_t err = cudaGetLastError();
    const char* err_str = cudaGetErrorString(err);
    TORCH_CHECK(err == cudaSuccess, std::string("CUDA kernel launch failed: ") + std::string(err_str));
    return torch::cat({V_out, D_out}, 2);
}

PYBIND11_MODULE(_core, m) {
    m.def("forward", &hsru_forward, "HSRU forward (CUDA, strict isolation)");
}