from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SurrogateSpike(Function):
    """
    A surrogate gradient function for the discrete spiking mechanism.

    Forward Pass: A Heaviside step function, producing a binary spike (0 or 1).
    Backward Pass: A smooth, bell-shaped surrogate gradient is substituted to
                   enable gradient-based learning.
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input_tensor, = ctx.saved_tensors
        # Surrogate is the derivative of a fast sigmoid function.
        surrogate_grad = (1 / (1 + 10 * torch.abs(input_tensor))).pow(2)
        return grad_output * surrogate_grad


spike_fn = SurrogateSpike.apply

class DualStateLIFLayer(nn.Module):
    """
    A recurrent cell with a dual-state memory system.

    It maintains:
    1. An Analog State (V): A continuous, leaky integrator for signal processing.
    2. A Digital "Mirror" State (D): A binary state for tracking discrete events.

    Args:
        input_size (int): The number of features in the input at each timestep.
        hidden_size (int): The number of neurons in the layer.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.fc_out = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.output_activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor], return_spike: bool = False) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        V_prev, D_prev = (s.detach() for s in state)
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))
        input_current = self.linear_in_v(x_t)
        V_t = leak_alpha * V_prev + input_current
        spike = spike_fn(V_t - self.flip_threshold)
        D_t = D_prev * (1 - spike) + (1 - D_prev) * spike
        combined_state = torch.cat([V_t, D_t], dim=1)
        output = self.output_activation(self.fc_out(combined_state))
        output = output if not return_spike else D_t
        return output, (V_t, D_t)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.flip_threshold.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        D0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return V0, D0

class HSRnn(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.rnn_cells = nn.ModuleList()
        layer_input_size = input_size
        for _ in range(num_layers):
            self.rnn_cells.append(DualStateLIFLayer(layer_input_size, hidden_size))
            layer_input_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        device = x.device

        states = [cell.init_state(batch_size, device) for cell in self.rnn_cells]
        x_t = x
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                x_t, new_state = cell(x_t, states[i])
                states[i] = new_state

        return x_t


class HSRnnForCausalLM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int]):
        super().__init__()
        self.rnn_cells = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.rnn_cells.append(DualStateLIFLayer(layer_input_size, hidden_size))
            layer_input_size = hidden_size

        self.lm_head = nn.Linear(hidden_layers_config[-1], output_size)

    def forward(self, x: torch.Tensor, return_spike: bool = False) -> torch.Tensor:
        """Processes a sequence and returns the output of the final timestep."""
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [cell.init_state(batch_size, device) for cell in self.rnn_cells]
        outputs_over_time = []
        for t in range(sequence_length):
            output = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                output, new_state = cell(output, states[i], return_spike)
                states[i] = new_state
            outputs_over_time.append(output)
        output_sequence = torch.stack(outputs_over_time, dim=1)
        logits = self.lm_head(output_sequence)
        return logits