#include "hsru_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void hsru_forward_kernel_impl(
    const float* __restrict__ input_seq,
    const float* __restrict__ leak,
    const float* __restrict__ threshold,
    float* __restrict__ V_out,
    float* __restrict__ D_out,
    const int batch_size,
    const int seq_len,
    const int hidden_size)
{
    const int b = blockIdx.x;
    const int h = threadIdx.x;
    if (h >= hidden_size) return;

    float V_prev = 0.0f;
    float D_prev = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        // Calculate the index for the current (b, t, h)
        const int idx = b * seq_len * hidden_size + t * hidden_size + h;

        // The recurrent update logic is the same, but simpler
        const float V_t = leak[h] * V_prev + input_seq[idx];
        const float spike = (V_t > threshold[h]) ? 1.0f : 0.0f;
        const float D_t = D_prev * (1.0f - spike) + (1.0f - D_prev) * spike;

        // Write outputs
        V_out[idx] = V_t;
        D_out[idx] = D_t;

        // Update state for the next time step
        V_prev = V_t;
        D_prev = D_t;
    }
}


// Implementation of the pure C launcher function
extern "C" void launch_hsru_kernel(
    const float* input_seq,
    const float* leak,
    const float* threshold,
    float* V_out,
    float* D_out,
    const int batch_size,
    const int seq_len,
    const int hidden_size)
{
    const dim3 blocks(batch_size);
    const dim3 threads(hidden_size);

    hsru_forward_kernel_impl<<<blocks, threads>>>(
        input_seq, leak, threshold,
        V_out, D_out,
        batch_size, seq_len, hidden_size);
}