# test.py

import os
import sys
import torch

# This block is still essential and correct.
if sys.platform == 'win32':
    cuda_home = os.environ.get('CUDA_HOME', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1')
    cuda_bin_path = os.path.join(cuda_home, 'bin')
    if os.path.exists(cuda_bin_path):
        os.add_dll_directory(cuda_bin_path)

try:
    import hsru_cuda_kernel

    print("SUCCESS: Package 'hsru_cuda_kernel' imported.")
    print("Contents of the package:", dir(hsru_cuda_kernel))
    assert 'forward' in dir(hsru_cuda_kernel), "'forward' function not found in package!"
    print("SUCCESS: The 'forward' function is available.")
    print("Testing the forward function...")
    device = 'cuda'
    input_t = torch.randn(4, 10, 64, device=device)
    leak_t = torch.rand(64, device=device)
    thresh_t = torch.rand(64, device=device)
    output = hsru_cuda_kernel.forward(input_t, leak_t, thresh_t)
    print("\nSUCCESS! The forward pass executed without errors.")
    print(f"Output shape: {output.shape}")
    print(f"Output tensor device: {output.device}")

except Exception as e:
    print(f"\nTEST FAILED.")
    print(f"Error details: {e}")
