import torch
import torch.nn as nn
import time

from core.hsru import HSRnn


def time_model(model, x, device_name):
    # Warmup
    for _ in range(10):
        _ = model(x)

    # Timing with CUDA sync
    if torch.cuda.is_available() and 'cuda' in device_name:
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(100):
        _ = model(x)

    if torch.cuda.is_available() and 'cuda' in device_name:
        torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / 100


# Experiment Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 64
hidden_size = 256
batch_size = 128
sequence_length = 500  # A reasonably long sequence

x = torch.randn(batch_size, sequence_length, input_size).to(device)

# Instantiate the models
slow_model = HSRnn(input_size, hidden_size).to(device)
# Instantiate first, THEN script
fast_model_jit = torch.jit.script(slow_model)

# Run Benchmark
slow_time = time_model(slow_model, x, str(device))
fast_time = time_model(fast_model_jit, x, str(device))

print(f"Running on: {device}")
print(f"Sequence Length: {sequence_length}")
print(f"Slow RNN (Python Loop): {slow_time * 1000:.4f} ms per forward pass")
print(f"Fast RNN (JIT Script):  {fast_time * 1000:.4f} ms per forward pass")
print("-" * 30)
print(f"Speedup: {slow_time / fast_time:.2f}x")