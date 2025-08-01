# hsru_parity_test.py

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
# --- Step 1: Import Your Custom Kernel ---
# This block ensures that Python can find the necessary CUDA DLLs on Windows.
if sys.platform == 'win32':
    cuda_home = os.environ.get('CUDA_HOME', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1')
    cuda_bin_path = os.path.join(cuda_home, 'bin')
    if os.path.exists(cuda_bin_path):
        os.add_dll_directory(cuda_bin_path)

# Import the forward function from your compiled package
try:
    from hsru_cuda_kernel import forward as hsru_forward_cuda

    print("Successfully imported custom CUDA kernel.")
except ImportError as e:
    print(f"FATAL: Could not import the compiled kernel. Please ensure it's installed correctly.")
    print(f"Error: {e}")
    sys.exit(1)


# --- Step 2: Define the RNN Layer that USES your kernel ---
class ParallelHSRnnV2(nn.Module):
    """
    This is the nn.Module that wraps the call to our custom CUDA kernel.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # We still define PyTorch layers for parts of the logic that are not in the kernel,
        # like the input projection and final output layer.
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.fc_out = nn.Linear(hidden_size * 2, hidden_size)  # V and D are concatenated
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("Model and data must be on the GPU to use the CUDA kernel.")

        # 1. Non-recurrent parts run as standard PyTorch layers
        input_currents = self.linear_in_v(x)
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))

        # 2. Call our super-fast, custom CUDA kernel for the recurrent part!
        combined_state = hsru_forward_cuda(
            input_currents,
            leak_alpha,
            self.flip_threshold
        )

        # 3. The final output layer is also a standard PyTorch layer
        output_sequence = self.output_activation(self.fc_out(combined_state))

        return output_sequence


# --- Step 3: Define the Full Classifier Model ---
class ParityClassifier(nn.Module):
    """Wraps our custom RNN and adds a classification head."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = ParallelHSRnnV2(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Get hidden states for the entire sequence from our custom RNN
        hidden_states = self.rnn(x)

        # We only care about the state after the last element is processed
        final_hidden_state = hidden_states[:, -1, :]

        # Classify based on this final state
        output = self.classifier(final_hidden_state)
        return output


# --- Step 4: Data Generation ---
def create_parity_data(num_samples, seq_len):
    """Generates binary sequences and their parity labels."""
    X = torch.randint(0, 2, (num_samples, seq_len, 1)).float()
    s = torch.sum(X, dim=1)
    y = s % 2
    return X, y.float()


# --- Step 5: Training and Evaluation Script ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("This test requires a CUDA-enabled GPU.")
        sys.exit(0)

    device = 'cuda'

    # Hyperparameters
    SEQ_LEN = 30
    INPUT_SIZE = 1
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1  # A single logit for binary classification
    LEARNING_RATE = 0.005
    BATCH_SIZE = 128
    EPOCHS = 15

    # Create the model, loss function, and optimizer
    model = ParityClassifier(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Generate Data
    print("Generating training and test data...")
    X_train, y_train = create_parity_data(20000, SEQ_LEN)
    X_test, y_test = create_parity_data(2000, SEQ_LEN)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Training Loop
    print(f"\n--- Starting Training for {EPOCHS} Epochs on {device} ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # Move data to the GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Avg Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation Loop
    print("\n--- Starting Evaluation ---")
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move data to the GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            predicted = (outputs.squeeze() > 0).float()
            total_correct += (predicted == y_batch.squeeze()).sum().item()
            total_samples += y_batch.size(0)

    accuracy = 100 * total_correct / total_samples
    print(f"\nAccuracy on the test set: {accuracy:.2f}%")

    if accuracy > 98:
        print("\n🎉 SUCCESS! Your custom CUDA kernel and model architecture have learned the parity task!")
    else:
        print("\nModel trained, but accuracy is low. Further hyperparameter tuning may be needed.")
