import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from core.hsru_parallal import HSRnn
from tests.wrapper import RNNClassifier, LSTMExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def run_single_trial(model: nn.Module, train_loader, val_loader, epochs, lr):
    """Runs a single training and evaluation trial for a given model."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        epoch_accuracy = accuracy_score(all_true, all_preds)
        val_accuracies.append(epoch_accuracy)

    return val_accuracies


def find_best_config(model_name: str, model_class: nn.Module, model_args: dict,
                     train_loader, val_loader, epochs: int, lr_space: List[float]):
    """Performs a hyperparameter sweep to find the best LR for a model."""
    print(f"\n--- Finding Best Config for: {model_name} ---")
    best_accuracy = 0
    best_lr = None
    best_curve = None

    for lr in lr_space:
        print(f"  Testing LR = {lr}...")
        set_seed(42)
        model = model_class(**model_args)
        learning_curve = run_single_trial(model, train_loader, val_loader, epochs, lr)
        final_accuracy = learning_curve[-1]
        print(f" -> Final Accuracy: {final_accuracy:.2%}")

        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_lr = lr
            best_curve = learning_curve

    final_model = model_class(**model_args)
    num_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"  > Optimal LR: {best_lr} | Peak Accuracy: {best_accuracy:.2%} | Params: {num_params:,}")

    return best_accuracy, best_lr, best_curve, num_params


def run_final_benchmark():
    """Main function to orchestrate the full, fair benchmark."""
    print("Setting up Final Benchmark")
    set_seed(42)
    INPUT_SIZE = 1
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    NUM_CLASSES = 2
    SEQ_LEN = 50
    BATCH_SIZE = 256
    EPOCHS = 40

    print(f"Running on device: {device}")

    print("Generating Temporal Parity Task data...")
    num_samples = BATCH_SIZE * 50
    X_data = torch.randint(0, 2, (num_samples, SEQ_LEN, INPUT_SIZE)).float()
    y_data = (X_data.sum(dim=(1, 2)) % 2 == 1).long()
    dataset = torch.utils.data.TensorDataset(X_data, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    learning_rate_space = [5e-3, 1e-3, 5e-4, 1e-4]

    lstm_backbone_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    lstm_classifier_args = {'rnn_backbone': LSTMExtractor(**lstm_backbone_args), 'hidden_size': HIDDEN_SIZE,
                            'num_classes': NUM_CLASSES}
    lstm_acc, lstm_lr, lstm_curve, lstm_params = find_best_config(
        "LSTM (Baseline)", RNNClassifier, lstm_classifier_args, train_loader, val_loader, EPOCHS, learning_rate_space
    )

    hsru_rnn_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    hsru_classifier_args = {'rnn_backbone': HSRnn(**hsru_rnn_args), 'hidden_size': HIDDEN_SIZE,
                            'num_classes': NUM_CLASSES}
    hsru_acc, hsru_lr, hsru_curve, hsru_params = find_best_config(
        "HSRnn", RNNClassifier, hsru_classifier_args, train_loader, val_loader, EPOCHS, learning_rate_space
    )

    print("\n\n" + "=" * 40 + " FINAL BENCHMARK SUMMARY " + "=" * 40)
    print(f"| {'Model':<20} | {'Peak Accuracy':<15} | {'Optimal LR':<15} | {'Parameters':<15} |")
    print(f"|{'-' * 22}|{'-' * 17}|{'-' * 17}|{'-' * 17}|")
    print(f"| {'LSTM (Baseline)':<20} | {f'{lstm_acc:.2%}':<15} | {lstm_lr:<15} | {f'{lstm_params:,}':<15} |")
    print(f"| {'HSRnn':<20} | {f'{hsru_acc:.2%}':<15} | {hsru_lr:<15} | {f'{hsru_params:,}':<15} |")
    print("=" * 81)

    plt.figure(figsize=(12, 7))
    plt.plot(range(1, EPOCHS + 1), lstm_curve, marker='s', linestyle='--',
             label=f'LSTM (Best LR={lstm_lr}) - Final Acc: {lstm_acc:.2%}')
    plt.plot(range(1, EPOCHS + 1), hsru_curve, marker='o', linestyle='-',
             label=f'HSRnn (Best LR={hsru_lr}) - Final Acc: {hsru_acc:.2%}')
    plt.title("HSRnn vs. LSTM on Temporal Parity Task (Optimal Configurations)", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0.45, top=1.02)
    plt.show()


if __name__ == '__main__':
    run_final_benchmark()
