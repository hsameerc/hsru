# tests/verify_data.py

import numpy as np
from collections import Counter

import torch


def generate_parity_data(n_samples, seq_len, input_size=1):
    """Generates a batch of data for the Temporal Parity Task."""
    X = torch.randint(0, 2, (n_samples, seq_len, input_size)).float()
    y = (X.sum(dim=(1, 2)) % 2 == 1).long()
    return X, y


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

