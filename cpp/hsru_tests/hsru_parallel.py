import torch
import torch.nn as nn
import torch.nn.functional as F

import hsru_cuda_kernel


class ParallelHSRnnV2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.fc_out = nn.Linear(hidden_size * 2, hidden_size)  # V and D are concatenated
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the correct device
        if not x.is_cuda:
            raise RuntimeError("Model and data must be on the GPU to use the CUDA kernel.")

        input_currents = self.linear_in_v(x)
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))

        # The call is the same, but now it works for ANY hidden_size!
        combined_state = hsru_cuda_kernel.forward(
            input_currents,
            leak_alpha,
            self.flip_threshold
        )

        output_sequence = self.output_activation(self.fc_out(combined_state))
        return output_sequence


# Example test:
if __name__ == '__main__':
    # Try with different hidden sizes!
    HIDDEN_SIZE_1 = 32
    HIDDEN_SIZE_2 = 96  # This would have failed before

    model_1 = ParallelHSRnnV2(1, HIDDEN_SIZE_1).to('cuda')
    model_2 = ParallelHSRnnV2(1, HIDDEN_SIZE_2).to('cuda')

    test_input = torch.randn(16, 50, 1).to('cuda')

    print(f"Testing with hidden_size={HIDDEN_SIZE_1}...")
    output_1 = model_1(test_input)
    print(f"Output shape: {output_1.shape}")  # Should be (16, 50, 32)
    print("Success!")

    print(f"\nTesting with hidden_size={HIDDEN_SIZE_2}...")
    output_2 = model_2(test_input)
    print(f"Output shape: {output_2.shape}")  # Should be (16, 50, 96)
    print("Success!")
