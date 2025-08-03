# HSRU: Hybrid State Recurrent Unit

A PyTorch implementation of a Hybrid State Recurrent Unit (HSRU), a custom multi-layer RNN architecture inspired by biological neurons. This model is designed for sequence modeling and includes a high-performance, JIT-compiled implementation for efficient training.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

## Overview

The HSRU is a recurrent neural network built by stacking custom `DualStateLIFLayer` cells. Each cell maintains two distinct internal states, mimicking the membrane potential and refractory state of a Leaky-Integrate-and-Fire (LIF) neuron.

This hierarchical structure allows the model to learn representations at different temporal scales, where each layer processes the sequence of hidden states from the layer below it.

## Key Features

-   **Bio-inspired Dual-State Mechanism:** Each cell tracks both a continuous "voltage" (`V`) and a discrete "duration/spike" state (`D`), allowing for more complex temporal dynamics than a standard RNN/LSTM.
-   **Causal Language Model:** Includes a ready-to-use `HSRnnForCausalLM` wrapper for next-token prediction tasks.
-   **Customizable:** Easily configure the network's depth and hidden sizes.

## Architecture

The model consists of two main components:

### 1. `DualStateLIFLayer`

This is the fundamental building block. At each timestep `t`, it performs the following steps:
1.  **Input Integration:** It takes the input `x_t` and transforms it into an "input current".
2.  **Voltage Update:** The internal membrane potential `V_t` is updated based on the previous voltage `V_{t-1}` (with a trainable leak) and the new input current. This is a linear recurrence: `V_t = leak * V_{t-1} + input_current`.
3.  **Spiking:** If `V_t` crosses a trainable threshold, a "spike" is generated. This is a non-linear operation that gives the model its expressive power.
4.  **Duration State Update:** The second state, `D_t`, flips based on the spiking activity, modeling a refractory or activity-dependent state.
5.  **Output Generation:** The final output is computed from a combination of the voltage (`V_t`) and duration (`D_t`) states, passed through a final linear layer and a `Tanh` activation.

### 2. `HSRnnBase`

This module stacks multiple `DualStateLIFLayer` instances. It unrolls the computation over the time dimension. To achieve high performance, this module is JIT-compiled. It takes an entire sequence `x` and efficiently computes the output hidden states for every time step from the final layer.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/hsameerc/hsru.git
    cd hsru
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Your `requirements.txt` should contain:
    ```
    torch>=1.12.0
    ```

## Usage

The HSRU can be used as a general-purpose sequence model or specifically for causal language modeling.

### Example 1: Using the Base Model

This shows how to get hidden state sequences from the JIT-compiled backbone.

```python
import torch
from hsru.model import HSRnnBase # Assuming your model is in hsru/model.py

# Configuration
input_size = 128
hidden_layers_config = [256, 512] # Two layers with hidden sizes 256 and 512
batch_size = 32
sequence_length = 500

# Instantiate the model (this will JIT-compile it automatically)
model = HSRnnBase(input_size=input_size, hidden_layers_config=hidden_layers_config)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Create some random input data
x = torch.randn(batch_size, sequence_length, input_size, device=model.device)

# Forward pass
# This call is fast, as the internal loop runs in an optimized C++ graph.
output_sequence = model(x)

# Output shape will be (batch_size, sequence_length, last_hidden_size)
print("Output shape:", output_sequence.shape)
# Expected: torch.Size([32, 500, 512])
```

### Example 2: Causal Language Modeling

This is the primary use case, which adds a language modeling head on top of the backbone.

```python
import torch
from src.core.hsru_casual_lm import HSRnnForCausalLM

# Configuration
vocab_size = 10000
embedding_size = 128
hidden_layers_config = [256, 512]
batch_size = 32
sequence_length = 500

# Instantiate the full language model
lm_model = HSRnnForCausalLM(
    input_size=embedding_size,
    output_size=vocab_size,
    hidden_layers_config=hidden_layers_config
)
lm_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Create some random input data (e.g., from an embedding layer)
x = torch.randn(batch_size, sequence_length, embedding_size, device=lm_model.device)

# Get the logits for next-token prediction
logits = lm_model(x)

# Logits shape will be (batch_size, sequence_length, vocab_size)
print("Logits shape:", logits.shape)
# Expected: torch.Size([32, 500, 10000])
```

## Training Example

Here is a minimal training loop for the `HSRnnForCausalLM`.

```python
import torch
import torch.nn as nn
from src.core.hsru_casual_lm import HSRnnForCausalLM

# --- Model and Data Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 10000
embedding_size = 128
hidden_layers_config = [256, 512]
sequence_length = 500
batch_size = 16

model = HSRnnForCausalLM(embedding_size, vocab_size, hidden_layers_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# --- Dummy Data ---
# Input: sequence of token IDs
input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
# Target: next token in the sequence (shifted by one)
target_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
# A mock embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_size).to(device)


# --- Training Loop ---
model.train()
for epoch in range(5): # Loop over epochs
    # Get embeddings for the input
    inputs_embeds = embedding_layer(input_ids)

    # Forward pass
    logits = model(inputs_embeds) # Shape: (batch, seq_len, vocab_size)

    # Calculate loss
    # CrossEntropyLoss expects logits as (N, C, ...) and targets as (N, ...)
    # So we reshape our tensors
    loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## To-Do & Future Work
-   [ ] Provide detailed benchmarks against standard LSTM, GRU, and Transformer models.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citing

If you use HSRU in your research, please consider citing this repository:

```bibtex
@software{your_name_2025_hsru,
  author = {Sameer Humagain},
  title = {{HSRU: A Hybrid State Recurrent Unit}},
  month = {July},
  year = {2025},
  publisher = {GitHub},
  version = {1.0.0},
  url = {https://github.com/hsameerc/hsru}
}
```