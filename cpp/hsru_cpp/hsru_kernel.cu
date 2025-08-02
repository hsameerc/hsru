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
    const int b = blockIdx.x;           // Batch index
    const int start_h = threadIdx.x;    // Starting hidden unit index for this thread
    const int stride = blockDim.x;      // Stride for hidden unit loop

    for (int h = start_h; h < hidden_size; h += stride) {
        float V_prev = 0.0f;
        float D_prev = 0.0f;

        for (int t = 0; t < seq_len; ++t) {
            const int idx = b * seq_len * hidden_size + t * hidden_size + h;

            const float V_t = leak[h] * V_prev + input_seq[idx];
            const float spike = (V_t > threshold[h]) ? 1.0f : 0.0f;
            const float D_t = D_prev * (1.0f - spike) + (1.0f - D_prev) * spike;

            V_out[idx] = V_t;
            D_out[idx] = D_t;

            V_prev = V_t;
            D_prev = D_t;
        }
    }
}

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
    const int max_threads_per_block = 1024;
    const int threads = std::min(hidden_size, max_threads_per_block);
    const dim3 blocks(batch_size);
    const dim3 threads_per_block(threads);

    hsru_forward_kernel_impl<<<blocks, threads_per_block>>>(
        input_seq, leak, threshold,
        V_out, D_out,
        batch_size, seq_len, hidden_size);
}
