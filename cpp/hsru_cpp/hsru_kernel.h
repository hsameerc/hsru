// hsru_cpp/hsru_kernel.h

#pragma once
// Use extern "C" to prevent C++ name mangling, creating a clean C-style ABI.
#ifdef __cplusplus
extern "C" {
#endif

void launch_hsru_kernel(
    const float* input_seq,
    const float* leak,
    const float* threshold,
    float* V_out,
    float* D_out,
    const int batch_size,
    const int seq_len,
    const int hidden_size
);

#ifdef __cplusplus
}
#endif