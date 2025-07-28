from typing import List

import torch
import torch.nn as nn

from core.hsru import DualStateLIFLayer


# ==============================================================================
# The "Base Model" - Sequence-to-Vector Feature Extractor
# ==============================================================================
class HSRnnBase(nn.Module):
    """
    The base HSRnn model, designed as a pure feature extractor.

    This module processes a sequence and returns only the **final hidden state**
    of the last layer. It is a sequence-to-vector model, ideal for tasks like
    classification or as a memory encoder that produces a single summary vector.

    Args:
        input_size (int): The dimension of the input embeddings.
        hidden_layers_config (List[int]): A list defining the hidden size of each layer.
        **kwargs: Additional arguments for the HSRULayer.
    """

    def __init__(self, input_size: int, hidden_layers_config: List[int]):
        super().__init__()
        self.rnn_cells = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.rnn_cells.append(DualStateLIFLayer(layer_input_size, hidden_size))
            layer_input_size = hidden_size

        self.last_hidden_size = hidden_layers_config[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a sequence and returns only the final hidden state.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            A tensor representing the final hidden state of the last layer, of shape
            (batch_size, last_hidden_size).
        """
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [cell.init_state(batch_size, device) for cell in self.rnn_cells]

        x_t = x
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                # x_t is updated in-place as it passes through the layers
                x_t, new_state = cell(x_t, states[i])
                states[i] = new_state

        return x_t


# ==============================================================================
# The "Model with Head" - The Complete Language Model (Sequence-to-Sequence)
# ==============================================================================

class HSRnnForCausalLM(nn.Module):
    """
    A Causal Language Model built using the HSRnnBase.

    This model is a sequence-to-sequence model. It processes a sequence and
    returns a prediction for the next token at every position.
    """

    def __init__(self,  input_size: int, output_size: int, hidden_layers_config: List[int]):
        super().__init__()
        self.backbone = HSRnnBase(
            input_size=input_size,
            hidden_layers_config=hidden_layers_config,
        )
        self.lm_head = nn.Linear(self.backbone.last_hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This forward pass must be different from the base model.
        It needs to collect the outputs at every timestep.
        """
        batch_size, sequence_length, _ = x.shape
        device = x.device

        states = [cell.init_state(batch_size, device) for cell in self.backbone.rnn_cells]

        outputs_over_time = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.backbone.rnn_cells):
                x_t, new_state = cell(x_t, states[i])
                states[i] = new_state
            outputs_over_time.append(x_t)

        output_sequence = torch.stack(outputs_over_time, dim=1)

        logits = self.lm_head(output_sequence)

        return logits


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    model_config = {
        'input_size': 16,
        'hidden_layers_config': [128, 64],
        'output_size': 1000,
    }

    print("--- Testing Base Model (HSRnnBase) ---")
    base_model = HSRnnBase(
        input_size=model_config['input_size'],
        hidden_layers_config=model_config['hidden_layers_config']
    )
    dummy_input = torch.randn(4, 50, 16)
    final_state = base_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Base model output shape (final hidden state): {final_state.shape}")
    assert final_state.shape == (4, 64)
    print("✅ HSRnnBase works as a Seq2Vec model.")

    print("\n--- Testing Full Language Model (HSRnnForCausalLM) ---")
    lm_model = HSRnnForCausalLM(**model_config)
    output_logits = lm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"LM model output shape (logits sequence): {output_logits.shape}")
    assert output_logits.shape == (4, 50, 1000)
    print("✅ HSRnnForCausalLM works as a Seq2Seq model.")
