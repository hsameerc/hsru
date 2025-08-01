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
    // A temporary workspace for states, allocated in shared memory if it fits,
    // otherwise it could be passed from global memory. For simplicity,
    // we'll use a stack-based approach here assuming hidden_size is small.
    // For large hidden_size, passing a workspace pointer is better.
    float V_prev[256]; // Example static size, must be >= hidden_size
    float D_prev[256];

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    
    // Initialize states
    for (int h = 0; h < hidden_size; ++h) {
        V_prev[h] = 0.0f; D_prev[h] = 0.0f;
    }

    // Pointers to data for this batch item
    const float* input_b = input_seq + b * seq_len * hidden_size;
    float* V_out_b = V_out + b * seq_len * hidden_size;
    float* D_out_b = D_out + b * seq_len * hidden_size;

    // Recurrent loop
    for (int t = 0; t < seq_len; ++t) {
        for (int h = 0; h < hidden_size; ++h) {
            const int idx = t * hidden_size + h;
            const float V_t = leak[h] * V_prev[h] + input_b[idx];
            const float spike = (V_t > threshold[h]) ? 1.0f : 0.0f;
            const float D_t = D_prev[h] * (1.0f - spike) + (1.0f - D_prev[h]) * spike;
            V_out_b[idx] = V_t;
            D_out_b[idx] = D_t;
            V_prev[h] = V_t;
            D_prev[h] = D_t;
        }
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
    const dim3 threads(256);
    const dim3 blocks((batch_size + threads.x - 1) / threads.x);

    hsru_forward_kernel_impl<<<blocks, threads>>>(
        input_seq, leak, threshold,
        V_out, D_out,
        batch_size, seq_len, hidden_size);
}