import random
import time
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm

import sys

sys.path.append('..')
try:
    from src.core.hsru import HSRnn
except ImportError:
    print("FATAL: Could not import HSRnn. Please check the path in sys.path.append().")


    class HSRnn(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, x): return torch.randn(x.shape[0], 64)


# ==============================================================================
# === 1. CONFIGURATION & SETUP
# ==============================================================================

class BenchmarkConfig:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model settings
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    # Training settings
    BATCH_SIZE = 512
    LEARNING_RATE = 2e-3
    GRAD_CLIP_VALUE = 1.0
    # Validation settings
    VAL_SAMPLES = 4096  # Use a large, stable validation set


def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# === 2. DATA GENERATION
# ==============================================================================

def generate_parity_data_tensors(n_samples, seq_len, device):
    """Generates a fresh, random batch of data for the parity task as PyTorch tensors."""
    X = torch.randint(0, 2, (n_samples, seq_len, 1), dtype=torch.float32, device=device)
    y = (X.sum(dim=(1, 2)) % 2 == 1).long()
    return X, y


# ==============================================================================
# === 3. MODEL DEFINITIONS
# ==============================================================================

class ParityClassifier(nn.Module):
    def __init__(self, rnn_backbone, hidden_size):
        super().__init__()
        self.rnn = rnn_backbone
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        features = self.rnn(x)
        return self.classifier(features), features


class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]


# ==============================================================================
# === 4. TRAINING & EVALUATION LOOP
# ==============================================================================

def train_and_eval_stage(model: nn.Module, val_data: (torch.Tensor, torch.Tensor), config: BenchmarkConfig,
                         stage_config: Dict) -> float:
    """Trains on fresh on-the-fly data and validates against a fixed set."""
    seq_len = stage_config['seq_len']
    epochs = stage_config['epochs']
    model_name = stage_config['model_name']
    title = f"{model_name} @ SeqLen={seq_len}"

    model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    X_val_fixed, y_val_fixed = val_data

    # TRAINING LOOP (On-the-fly data)
    pbar = tqdm(range(epochs), desc=f"Training {title}", leave=False)
    for epoch in pbar:
        model.train()
        X_train, y_train = generate_parity_data_tensors(config.BATCH_SIZE, seq_len, config.DEVICE)

        pred, _ = model(X_train)
        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_VALUE)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out_logits, _ = model(X_val_fixed)
        predictions = torch.argmax(out_logits, dim=1)
        final_acc = accuracy_score(y_val_fixed.cpu().numpy(), predictions.cpu().numpy())

    print(f"✅ {title}: Final Accuracy on Fixed Val Set = {final_acc:.2%}")
    return final_acc


# ==============================================================================
# === 5. VISUALIZATION
# ==============================================================================

def plot_all_hidden_states(models_dict: Dict[str, nn.Module], seq_len: int, n_samples: int = 512):
    """Plots t-SNE of hidden states from each model in a single, comparative figure."""
    print("\n" + "=" * 30 + f" GENERATING t-SNE @ SeqLen={seq_len} " + "=" * 30)

    all_hidden_2d, all_labels = {}, {}
    global_min_max = {'xmin': np.inf, 'xmax': -np.inf, 'ymin': np.inf, 'ymax': -np.inf}

    for name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            X, y = generate_parity_data_tensors(n_samples, seq_len, BenchmarkConfig.DEVICE)
            _, hidden = model(X)

        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        hidden_2d = tsne.fit_transform(hidden.cpu().numpy())
        all_hidden_2d[name], all_labels[name] = hidden_2d, y.cpu().numpy()

        global_min_max['xmin'] = min(global_min_max['xmin'], hidden_2d[:, 0].min())
        global_min_max['xmax'] = max(global_min_max['xmax'], hidden_2d[:, 0].max())
        global_min_max['ymin'] = min(global_min_max['ymin'], hidden_2d[:, 1].min())
        global_min_max['ymax'] = max(global_min_max['ymax'], hidden_2d[:, 1].max())

    buffer = (global_min_max['xmax'] - global_min_max['xmin']) * 0.1

    fig, axes = plt.subplots(1, len(models_dict), figsize=(7 * len(models_dict), 7.5), squeeze=False)
    colors, class_labels = ['#FF6B6B', '#4ECDC4'], ['Parity 0 (Even)', 'Parity 1 (Odd)']
    fig.suptitle(f"t-SNE Visualization of Final Hidden States (SeqLen={seq_len})", fontsize=18, y=0.98)

    for i, (name, ax) in enumerate(zip(models_dict.keys(), axes.flatten())):
        hidden_2d, labels = all_hidden_2d[name], all_labels[name]
        for label_val, color, class_name in zip(np.unique(labels), colors, class_labels):
            idx = labels == label_val
            ax.scatter(hidden_2d[idx, 0], hidden_2d[idx, 1], color=color, label=class_name, alpha=0.8, edgecolors='k',
                       linewidths=0.5)

        ax.set_title(name, fontsize=14)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2" if i == 0 else "")
        ax.set_xlim(global_min_max['xmin'] - buffer, global_min_max['xmax'] + buffer)
        ax.set_ylim(global_min_max['ymin'] - buffer, global_min_max['ymax'] + buffer)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==============================================================================
# === 6. MAIN BENCHMARK ORCHESTRATION
# ==============================================================================

def run_memory_horizon_benchmark():
    config = BenchmarkConfig()
    set_seed(config.SEED)
    print("=" * 60)
    print("      MEASURING THE MEMORY HORIZON OF RNN ARCHITECTURES      ")
    print("=" * 60)
    print(f"Running on device: {config.DEVICE}\n")

    models_to_test_builders = {
        "LSTM": lambda: ParityClassifier(LSTMWrapper(1, config.HIDDEN_SIZE, config.NUM_LAYERS), config.HIDDEN_SIZE),
        "GRU": lambda: ParityClassifier(GRUWrapper(1, config.HIDDEN_SIZE, config.NUM_LAYERS), config.HIDDEN_SIZE),
        "HSRU": lambda: ParityClassifier(HSRnn(1, config.HIDDEN_SIZE, config.NUM_LAYERS), config.HIDDEN_SIZE),
    }

    curriculum = [
        {'seq_len': 10, 'epochs': 100},
        {'seq_len': 15, 'epochs': 150},
        {'seq_len': 20, 'epochs': 200},
        {'seq_len': 30, 'epochs': 250},
        {'seq_len': 50, 'epochs': 300},
    ]

    results = {name: [] for name in models_to_test_builders.keys()}
    sequence_lengths = [stage['seq_len'] for stage in curriculum]
    final_models = {}

    for stage in curriculum:
        seq_len = stage['seq_len']
        print("\n\n" + "#" * 30 + f" CURRICULUM STAGE: SeqLen={seq_len} " + "#" * 30)

        print(f"Generating fixed validation set for SeqLen={seq_len}...")
        X_val_fixed, y_val_fixed = generate_parity_data_tensors(config.VAL_SAMPLES, seq_len, config.DEVICE)
        val_data = (X_val_fixed, y_val_fixed)

        for name, model_builder in models_to_test_builders.items():
            model = model_builder()  # Re-initialize model for each stage
            stage['model_name'] = name
            final_acc = train_and_eval_stage(model, val_data, config, stage)
            results[name].append(final_acc)
            if seq_len == curriculum[-1]['seq_len']:  # Save final models
                final_models[name] = model

    print("\n\n" + "=" * 30 + " MEMORY HORIZON BENCHMARK SUMMARY " + "=" * 30)
    header = f"| {'Seq Len':<10} |" + "".join([f" {name:<15} |" for name in models_to_test_builders.keys()])
    print(header)
    print(f"|{'-' * 12}|" + "".join([f"{'-' * 17}|" for _ in models_to_test_builders.keys()]))
    for i, seq_len in enumerate(sequence_lengths):
        row = f"| {seq_len:<10} |"
        for name in models_to_test_builders.keys():
            row += f" {results[name][i]:<15.2%} |"
        print(row)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    for name, acc_list in results.items():
        plt.plot(sequence_lengths, acc_list, marker='o', linestyle='-', label=name)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess (50%)')
    plt.title("RNN Memory Horizon on the Temporal Parity Task", fontsize=16)
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Final Validation Accuracy", fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=11)
    plt.ylim(0.45, 1.05)
    plt.grid(True, which="both", linestyle='--')
    plt.show()

    plot_all_hidden_states(final_models, curriculum[-1]['seq_len'])


if __name__ == "__main__":
    run_memory_horizon_benchmark()