# tests/verify_data.py

import numpy as np
from collections import Counter

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def generate_parity_tensors(n_samples, seq_len, input_size=1):
    """Generates a batch of data for the Temporal Parity Task."""
    X = torch.randint(0, 2, (n_samples, seq_len, input_size)).float()
    y = (X.sum(dim=(1, 2)) % 2 == 1).long()
    return X, y


def prepare_parity_dataloaders(seq_len, input_size=1, batch_size=256, split_ratio=0.8, total_batches=50, device='cpu'):
    """
    Generates a full dataset for the Temporal Parity Task and returns train/val DataLoaders.

    Args:
        seq_len (int): Length of each binary sequence.
        input_size (int): Input feature size (default 1 for binary).
        batch_size (int): Batch size for loaders.
        split_ratio (float): Fraction of data to use for training.
        total_batches (int): Number of batches to generate.
        device (str or torch.device): Device to move tensors to.

    Returns:
        train_loader, val_loader (DataLoader): PyTorch DataLoaders for training and validation.
    """
    print("📦 Generating Temporal Parity Task dataset...")
    num_samples = batch_size * total_batches
    X_data, y_data = generate_parity_tensors(num_samples, seq_len, input_size)
    X_data, y_data = X_data.to(device), y_data.to(device)

    dataset = TensorDataset(X_data, y_data)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"✅ Dataset ready: {train_size} train samples, {val_size} val samples.")
    return train_loader, val_loader



def generate_parity_sequences(num_sequences, sequence_length):
    """Generates binary sequences and their parity labels as lists."""
    sequences = []
    labels = []
    for _ in range(num_sequences):
        seq = np.random.randint(0, 2, size=sequence_length).tolist()
        label = sum(seq) % 2
        sequences.append(seq)
        labels.append(label)
    return sequences, labels


def create_rigorous_parity_datasets(train_samples, val_samples, seq_len):
    """
    Generates perfectly non-overlapping train and validation sets.

    This is the gold standard for preventing data leakage in benchmarks.
    """
    print(f"\nGenerating a pool of {train_samples + val_samples:,} unique sequences...")

    # Use a set to automatically handle and store only unique sequences
    sequence_pool = set()

    # To be safe, set a max number of attempts to prevent infinite loops
    # if the desired number of samples is close to the total possible unique sequences.
    max_attempts = (train_samples + val_samples) * 5
    attempts = 0

    while len(sequence_pool) < (train_samples + val_samples) and attempts < max_attempts:
        # Generate sequences as tuples so they can be added to a set
        seq = tuple(np.random.randint(0, 2, size=seq_len).tolist())
        sequence_pool.add(seq)
        attempts += 1

    if len(sequence_pool) < (train_samples + val_samples):
        raise ValueError(f"Could not generate enough unique samples for seq_len={seq_len}. "
                         "This can happen if the number of samples is too close to 2^{seq_len}.")

    # Convert the set to a list for shuffling and splitting
    sequence_list = list(sequence_pool)
    np.random.shuffle(sequence_list)  # Shuffle for random distribution

    # Deterministically split the single pool
    train_sequences = [list(seq) for seq in sequence_list[:train_samples]]
    val_sequences = [list(seq) for seq in sequence_list[train_samples:]]

    print("Generated non-overlapping train and validation sets.")
    return train_sequences, val_sequences

def run_data_diagnostics(num_samples=100000, seq_len=200):
    """
    Runs a series of checks on the data generation process to ensure
    it is fair, balanced, and diverse.
    """
    print("=" * 50)
    print("      RUNNING DATA GENERATION DIAGNOSTICS      ")
    print("=" * 50)
    print(f"Generating {num_samples:,} samples with sequence length {seq_len}...\n")

    # Generate a large dataset
    sequences, labels = generate_parity_sequences(num_samples, seq_len)

    # Label Balance
    print("Verifying Label Balance")
    label_counts = Counter(labels)
    total_labels = len(labels)
    percent_even = (label_counts[0] / total_labels) * 100
    percent_odd = (label_counts[1] / total_labels) * 100

    print(f"  - Total samples: {total_labels:,}")
    print(f"  - Count of 'Even' (0) labels: {label_counts[0]:,}")
    print(f"  - Count of 'Odd' (1) labels: {label_counts[1]:,}")
    print(f"  - Distribution: {percent_even:.2f}% Even / {percent_odd:.2f}% Odd")
    if abs(percent_even - 50) < 1.0:
        print("  ✅ RESULT: Labels are well-balanced, as expected.\n")
    else:
        print("  ⚠️ WARNING: Labels appear to be skewed.\n")

    # Data Diversity
    print("Verifying Data Diversity")

    # Convert lists to tuples to make them hashable for the set
    unique_sequences = set(map(tuple, sequences))
    num_unique = len(unique_sequences)
    total_sequences = len(sequences)
    diversity_ratio = (num_unique / total_sequences) * 100

    print(f"  - Total sequences generated: {total_sequences:,}")
    print(f"  - Number of unique sequences: {num_unique:,}")
    print(f"  - Diversity Ratio (Unique / Total): {diversity_ratio:.2f}%")

    # For a long sequence length like 200, collisions should be virtually impossible.
    if diversity_ratio > 99.99:
        print("  ✅ RESULT: Data is highly diverse with negligible random collisions.\n")
    else:
        print("  ⚠️ WARNING: A significant number of duplicate sequences were generated.\n")

    print("=" * 50)
    print("            DIAGNOSTICS COMPLETE            ")
    print("=" * 50)


if __name__ == "__main__":
    run_data_diagnostics(num_samples=100000, seq_len=200)
