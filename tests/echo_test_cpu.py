import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from src.core.hsru_parallal import HSRnnForCausalLM
from src.core.hsru import HSRnnForCausalLM

from tests.wrapper import LSTM_Seq2Seq_Wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used device {device}")
def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_echo_data(n_samples: int, seq_len: int, input_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates data for the Delayed Echo Task."""
    phases = torch.rand(n_samples, 1, 1) * 2 * np.pi
    time_steps = torch.linspace(0, 5 * np.pi, seq_len).reshape(1, -1, 1)
    X = torch.sin(time_steps + phases)
    y = torch.zeros_like(X)
    y[:, 1:, :] = X[:, :-1, :]
    return X.to(device), y.to(device)


def train_and_eval_echo(model_name: str, model: nn.Module, epochs=50, batch_size=128, seq_len=5000, lr=1e-3):
    """Performs a full training and evaluation for the echo task."""
    print(f"\n--- Training {model_name} on the Delayed Echo Task ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    final_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        X_train, y_train = generate_echo_data(batch_size, seq_len)
        optimizer.zero_grad()
        pred_sequence = model(X_train)
        loss = criterion(pred_sequence, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_val, y_val = generate_echo_data(batch_size, seq_len)
                pred_val = model(X_val)
                val_loss = criterion(pred_val, y_val)
            final_val_loss = val_loss.item()
            print(f"  Epoch {epoch + 1}/{epochs}, Validation Loss: {final_val_loss:.6f}")

    print(f"✅ Training complete for {model_name}.")

    # Final Visualization
    model.eval()
    with torch.no_grad():
        X_test, y_test = generate_echo_data(1, seq_len)
        pred_test = model(X_test)

    plt.figure(figsize=(12, 6))
    plt.title(f"Delayed Echo Task Performance: {model_name}", fontsize=16)
    plt.xlabel("Timestep");
    plt.ylabel("Signal Value")
    plt.plot(X_test[0].cpu().numpy(), label='Input Signal (X)', color='black', linestyle=':')
    plt.plot(y_test[0].cpu().numpy(), label='Target Signal (y = X shifted)', color='blue', linewidth=2)
    plt.plot(pred_test[0].cpu().numpy(), label='Model Prediction', color='red', linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.savefig(f"../plots/echo_test_cpu/{model_name}-Delayed-Echo-Task-Performance.png")
    return final_val_loss


def run_echo_benchmark():
    """Main function to orchestrate the sanity check benchmark."""
    set_seed(42)

    # Configuration
    INPUT_SIZE = 1
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1

    # Baseline: A standard, known-good LSTM
    lstm_backbone = LSTM_Seq2Seq_Wrapper(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )

    # HSRU: The ProjectedHSRnn
    # Ensure it's the Seq2Seq version that returns the full sequence
    lif_backbone = HSRnnForCausalLM(
        input_size=INPUT_SIZE,
        output_size=HIDDEN_SIZE,  # The RNN's intermediate output is its hidden state
        hidden_layers_config=[HIDDEN_SIZE] * NUM_LAYERS
    )

    # Use a final projection layer for both models to map hidden state to output dimension
    models_to_test = {
        "LSTM (Baseline)": nn.Sequential(lstm_backbone, nn.Linear(HIDDEN_SIZE, INPUT_SIZE)),
        "HSRU": nn.Sequential(lif_backbone, nn.Linear(HIDDEN_SIZE, INPUT_SIZE))
    }

    # Running the Tests
    results = {}
    for name, model in models_to_test.items():
        final_loss = train_and_eval_echo(name, model)
        results[name] = final_loss

    # Report
    print("\n\n" + "=" * 30 + " DELAYED ECHO BENCHMARK SUMMARY " + "=" * 30)
    print(f"| {'Model':<20} | {'Final Validation MSE Loss':<30} |")
    print(f"|{'-' * 22}|{'-' * 32}|")
    for name, loss in results.items():
        print(f"| {name:<20} | {loss:<30.6f} |")
    print("=" * 58)
    print("\n### Interpretation:")
    print("A low MSE Loss (< 0.01) and a close match on the plot confirm that the model's")
    print("core recurrent mechanics are functional and trainable.")


if __name__ == '__main__':
    run_echo_benchmark()
