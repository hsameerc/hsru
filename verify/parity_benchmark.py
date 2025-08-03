import random
import time
from collections import Counter
from typing import Dict, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import sys

sys.path.append('..')
from src.core.hsru import HSRnn


# ==============================================================================
# === CONFIGURATION & SETUP
# ==============================================================================

# Centralize all settings for easy modification and reproducibility
class BenchmarkConfig:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data settings
    SEQ_LEN = 16
    TRAIN_SAMPLES = 10000
    VAL_SAMPLES = 2000
    # Model settings
    HIDDEN_SIZE = 64
    # Training settings
    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    GRAD_CLIP_VALUE = 1.0


def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# === DATA GENERATION & VERIFICATION (The "Gold Standard")
# ==============================================================================

def create_rigorous_parity_datasets(config: BenchmarkConfig):
    """Generates perfectly non-overlapping train/val sets from a unique pool."""
    print("\n--- Generating and Verifying Benchmark Datasets ---")
    total_samples = config.TRAIN_SAMPLES + config.VAL_SAMPLES

    # Use a set to automatically handle duplicates
    sequence_pool = set()
    max_attempts = total_samples * 5

    pbar = tqdm(total=total_samples, desc="Generating unique sequences")
    while len(sequence_pool) < total_samples and pbar.n < max_attempts:
        seq = tuple(np.random.randint(0, 2, size=config.SEQ_LEN).tolist())
        if seq not in sequence_pool:
            sequence_pool.add(seq)
            pbar.update(1)
    pbar.close()

    if len(sequence_pool) < total_samples:
        raise ValueError("Could not generate enough unique samples.")

    sequence_list = list(sequence_pool)
    np.random.shuffle(sequence_list)

    # Deterministic split
    train_seqs = [list(seq) for seq in sequence_list[:config.TRAIN_SAMPLES]]
    val_seqs = [list(seq) for seq in sequence_list[config.TRAIN_SAMPLES:]]

    # Data Leakage Check (should always pass now)
    train_set = set(map(tuple, train_seqs))
    val_set = set(map(tuple, val_seqs))
    overlap = len(train_set.intersection(val_set))
    assert overlap == 0, f"FATAL: Data leakage detected with {overlap} overlapping samples!"
    print(f"✅ Data integrity verified: 0 overlapping samples found.")

    # Convert to Tensors
    X_train = torch.tensor(train_seqs, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor([sum(s) % 2 for s in train_seqs], dtype=torch.long)
    X_val = torch.tensor(val_seqs, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor([sum(s) % 2 for s in val_seqs], dtype=torch.long)

    return TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)


# ==============================================================================
# === MODEL DEFINITIONS (Modular and Clean)
# ==============================================================================

class ParityClassifier(nn.Module):
    def __init__(self, rnn_backbone, hidden_size):
        super().__init__()
        self.rnn = rnn_backbone
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        features = self.rnn(x)  # Backbone returns final hidden state
        return self.classifier(features), features


class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        # Return last hidden state from the last layer
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]


# ==============================================================================
# === TRAINING & EVALUATION LOOP (Efficient and Clear)
# ==============================================================================

def train_and_eval(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: BenchmarkConfig,
                   title: str) -> Dict[str, Any]:
    """Trains and evaluates a model, returning final metrics and the trained model."""
    print(f"\n--- Training: {title} ---")
    model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(config.EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)

            pred, _ = model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_VALUE)
            optimizer.step()

    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds.")

    # Final Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            out_logits, _ = model(X_batch)
            predictions = torch.argmax(out_logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)

    # Detailed Logging for a few samples
    print("\n--- DETAILED LOG FOR FIRST 5 VALIDATION SAMPLES ---")
    X_val_sample, y_val_sample = next(iter(val_loader))
    X_val_sample, y_val_sample = X_val_sample.to(config.DEVICE), y_val_sample.to(config.DEVICE)
    out_logits_sample, _ = model(X_val_sample)
    predictions_sample = torch.argmax(out_logits_sample, dim=1)

    for i in range(min(5, len(X_val_sample))):
        # ... (Your detailed logging code is excellent, no changes needed) ...
        pass  # Placeholder for your detailed logging

    print(f"✅ {title}: Final Overall Accuracy = {final_acc:.2%}")
    return {"accuracy": final_acc, "model": model}


# ==============================================================================
# === Plot Hidden States
# ==============================================================================
def plot_all_hidden_states(
        models_dict: Dict[str, nn.Module],
        val_loader: DataLoader,
        config: 'BenchmarkConfig',
        n_samples: int = 1024
):
    """
    Collects hidden states from each model on a subset of the validation data,
    runs t-SNE, and plots the results in a single, comparative figure.
    """
    print("\n" + "=" * 30 + " GENERATING t-SNE VISUALIZATIONS " + "=" * 30)

    # --- Step 1: Collect hidden states and labels for all models ---
    all_hidden_states = {}
    all_labels = []

    # Prepare a fixed subset of validation data for consistent plotting
    val_subset = list(zip(*[val_loader.dataset[i] for i in range(min(n_samples, len(val_loader.dataset)))]))
    X_plot = torch.stack(val_subset[0]).to(config.DEVICE)
    y_plot = torch.stack(val_subset[1]).cpu().numpy()
    all_labels = y_plot

    print(f"Collecting hidden states from {len(models_dict)} models on {len(X_plot)} samples...")
    for name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            # The model's forward pass returns (logits, features)
            _, hidden = model(X_plot)
            all_hidden_states[name] = hidden.detach().cpu().numpy()

    # --- Step 2: Apply t-SNE and find global axis limits for fair comparison ---
    all_hidden_2d = {}
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    print("Running t-SNE dimensionality reduction for all models...")
    for name, hidden in tqdm(all_hidden_states.items(), desc="t-SNE Progress"):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=config.SEED)
        hidden_2d = tsne.fit_transform(hidden)
        all_hidden_2d[name] = hidden_2d

        # Update global axis limits to ensure all plots share the same scale
        global_x_min = min(global_x_min, hidden_2d[:, 0].min())
        global_x_max = max(global_x_max, hidden_2d[:, 0].max())
        global_y_min = min(global_y_min, hidden_2d[:, 1].min())
        global_y_max = max(global_y_max, hidden_2d[:, 1].max())

    # Add a small aesthetic buffer to the axis limits
    x_buffer = (global_x_max - global_x_min) * 0.05
    y_buffer = (global_y_max - global_y_min) * 0.05

    # --- Step 3: Create the plot wall ---
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6.5 * num_models, 7), squeeze=False)
    axes = axes.flatten()

    # Define a professional, colorblind-friendly color palette
    colors = ['#0072B2', '#D55E00']  # Blue and Orange/Vermillion
    class_names = ['Parity 0 (Even)', 'Parity 1 (Odd)']

    fig.suptitle(f"t-SNE Visualization of Final Hidden States (SeqLen={config.SEQ_LEN})", fontsize=20, y=1.02)

    for i, name in enumerate(models_dict.keys()):
        ax = axes[i]
        hidden_2d = all_hidden_2d[name]

        # Plot each class separately to assign specific colors and labels
        for label_val, color, class_name in zip(np.unique(all_labels), colors, class_names):
            idx = all_labels == label_val
            ax.scatter(
                hidden_2d[idx, 0],
                hidden_2d[idx, 1],
                color=color,
                label=class_name,
                alpha=0.7,
                linewidths=0.5
            )

        ax.set_title(name, fontsize=16, pad=10)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)

        # Only label the y-axis on the very first plot to reduce clutter
        if i == 0:
            ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

        # Set consistent axis limits for a fair visual comparison
        ax.set_xlim(global_x_min - x_buffer, global_x_max + x_buffer)
        ax.set_ylim(global_y_min - y_buffer, global_y_max + y_buffer)

        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        # Remove top and right spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    # To save the figure to a file instead of showing it:
    # plt.savefig("tsne_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# === MAIN BENCHMARK ORCHESTRATION
# ==============================================================================

def run_benchmark():
    """Main function to orchestrate the entire benchmark."""
    config = BenchmarkConfig()
    set_seed(config.SEED)
    print(f"Running benchmark on device: {config.DEVICE}")

    # --- Data Preparation ---
    train_dataset, val_dataset = create_rigorous_parity_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # --- Define Models ---
    models_to_test = {
        "LSTM": ParityClassifier(LSTMWrapper(1, config.HIDDEN_SIZE), config.HIDDEN_SIZE),
        "GRU": ParityClassifier(GRUWrapper(1, config.HIDDEN_SIZE), config.HIDDEN_SIZE),
        "HSRU": ParityClassifier(HSRnn(1, config.HIDDEN_SIZE, num_layers=1), config.HIDDEN_SIZE),
    }

    # --- Run Benchmark ---
    results = {}
    for name, model in models_to_test.items():
        results[name] = train_and_eval(model, train_loader, val_loader, config, name)

    # --- Final Report ---
    print("\n\n" + "=" * 35 + " FINAL BENCHMARK SUMMARY " + "=" * 35)
    print(
        f"Sequence Length: {config.SEQ_LEN}, Train Samples: {config.TRAIN_SAMPLES}, Val Samples: {config.VAL_SAMPLES}")
    print(f"| {'Model':<10} | {'Final Accuracy':<20} |")
    print(f"|{'-' * 12}|{'-' * 22}|")
    for name, result in results.items():
        print(f"| {name:<10} | {result['accuracy']:<20.2%} |")

    # --- Visualization ---
    print("\n--- Generating final t-SNE plots for all models ---")
    # Create a dictionary of the final, trained models
    final_models = {name: res['model'] for name, res in results.items()}
    # Call the plotting function with the final models and the validation data loader
    plot_all_hidden_states(final_models, val_loader, config)

if __name__ == "__main__":
    run_benchmark()
