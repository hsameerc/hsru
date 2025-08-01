import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


if sys.platform == 'win32':
    # Attempt to get CUDA_HOME from environment variables, with a default fallback
    cuda_home = os.environ.get('CUDA_HOME', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1')
    cuda_bin_path = os.path.join(cuda_home, 'bin')
    if os.path.exists(cuda_bin_path):
        os.add_dll_directory(cuda_bin_path)

try:
    # Import the 'forward' function from your compiled package
    from hsru_cuda_kernel import forward as hsru_forward_cuda

    print("Successfully imported custom HSRU CUDA kernel.")
except ImportError as e:
    print("FATAL: Could not import the compiled 'hsru_cuda_kernel'.")
    print("Please ensure the package was installed correctly (e.g., 'pip install -e .')")
    print(f"Import Error: {e}")
    hsru_forward_cuda = None  # Set to None to cause an explicit error if used
    # sys.exit(1) # Uncomment to make a hard exit


class ParallelHSRULayer(nn.Module):
    """
    A high-performance recurrent layer that processes an entire sequence at once
    using a custom CUDA kernel.

    This layer replaces the Python for-loop over the time dimension with a single,
    highly optimized C++/CUDA function call.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if hsru_forward_cuda is None:
            raise RuntimeError("HSRU CUDA kernel is not available.")

        self.hidden_size = hidden_size

        # These are the trainable parameters for the recurrent logic
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))

        # This layer processes the combined output of the recurrent states
        self.fc_out = nn.Linear(hidden_size * 2, hidden_size)  # V and D are concatenated
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the entire input sequence in one go.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch, sequence_length, hidden_size)
        """
        if not x.is_cuda:
            raise RuntimeError("Input tensor must be on a CUDA device to use the HSRU CUDA kernel.")

        # Non-recurrent parts can be standard PyTorch operations
        input_currents = self.linear_in_v(x)
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))

        # The entire recurrent computation over the time dimension
        # is replaced by a single call to our custom CUDA kernel.
        combined_state_sequence = hsru_forward_cuda(
            input_currents,
            leak_alpha,
            self.flip_threshold
        )

        # The final output layer processes the entire sequence of states
        output_sequence = self.output_activation(self.fc_out(combined_state_sequence))

        return output_sequence


class HSRnn(nn.Module):
    """
    A multi-layer stacked HSRU model built with high-performance parallel layers.

    The slow Python for-loop over the sequence length is eliminated. The loop
    over the number of layers remains, which is efficient as num_layers is typically small.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.rnn_layers = nn.ModuleList()
        current_size = input_size
        for _ in range(num_layers):
            self.rnn_layers.append(ParallelHSRULayer(current_size, hidden_size))
            current_size = hidden_size  # The output of one layer is the input to the next

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the sequence through all layers.
        Returns the output of the final layer at the last timestep.
        """
        # The input 'x' is the full sequence (B, L, C)
        output_sequence = x

        # Loop over layers (efficient)
        for layer in self.rnn_layers:
            output_sequence = layer(output_sequence)

        # Return only the output from the final timestep
        return output_sequence[:, -1, :]


class HSRnnForCausalLM(nn.Module):
    """
    A high-performance HSRU model for causal language modeling tasks.

    This model processes the entire sequence in parallel and returns logits
    for every timestep, making it suitable for training with a standard
    cross-entropy loss on language tasks.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int]):
        super().__init__()
        self.rnn_layers = nn.ModuleList()
        current_size = input_size
        for hidden_size in hidden_layers_config:
            self.rnn_layers.append(ParallelHSRULayer(current_size, hidden_size))
            current_size = hidden_size

        # The final head to project hidden states to vocabulary logits
        self.lm_head = nn.Linear(hidden_layers_config[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a sequence and returns the logits for every timestep.
        """
        output_sequence = x

        # Passing the full sequence through the stack of fast RNN layers
        for layer in self.rnn_layers:
            output_sequence = layer(output_sequence)

        # Applying the language modeling head to the entire output sequence
        logits = self.lm_head(output_sequence)

        return logits


# Example Usage
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
    else:
        device = 'cuda'
        batch_size = 8
        seq_len = 100
        input_features = 64
        hidden_size = 128
        vocab_size = 1000

        # Create dummy input data
        dummy_input = torch.randn(batch_size, seq_len, input_features).to(device)

        print("--- Testing HSRnn ---")
        hsrnn_model = HSRnn(input_features, hidden_size, num_layers=2).to(device)
        output = hsrnn_model(dummy_input)
        print(f"HSRnn input shape: {dummy_input.shape}")
        print(f"HSRnn output shape (last timestep): {output.shape}")
        assert output.shape == (batch_size, hidden_size)
        print("HSRnn test successful.\n")

        print("--- Testing HSRnnForCausalLM ---")
        lm_config = [128, 256]  # Two layers with 128 and 256 hidden units
        lm_model = HSRnnForCausalLM(input_features, vocab_size, lm_config).to(device)
        logits = lm_model(dummy_input)
        print(f"HSRnnForCausalLM input shape: {dummy_input.shape}")
        print(f"HSRnnForCausalLM output shape (full sequence logits): {logits.shape}")
        assert logits.shape == (batch_size, seq_len, vocab_size)
        print("HSRnnForCausalLM test successful.")