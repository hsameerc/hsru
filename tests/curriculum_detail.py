import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

from core.hsru import HSRnn  # Assuming your HSRnn is in this file
from tests.wrapper import RNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParityClassifier(nn.Module):
    """A clean classifier wrapper for any recurrent backbone."""

    def __init__(self, rnn_backbone, hidden_size, output_size=2):
        super().__init__()
        self.rnn = rnn_backbone
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        final_hidden_state = self.rnn(x)
        logits = self.fc(final_hidden_state)
        return logits, final_hidden_state


class LSTMExtractor(nn.Module):
    """A wrapper for nn.LSTM to match our RNN interface (returns last hidden state)."""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (final_hidden_state, _) = self.lstm(x)
        return final_hidden_state[-1]


def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False;
    torch.backends.cudnn.deterministic = True


def generate_parity_data(n_samples, seq_len, input_size=1):
    """Generates a batch of data for the Temporal Parity Task."""
    X = torch.randint(0, 2, (n_samples, seq_len, input_size)).float()
    y = (X.sum(dim=(1, 2)) % 2 == 1).long()
    return X.to(device), y.to(device)


def run_single_trial(model: nn.Module, train_loader, val_loader, epochs, lr):
    """Runs a single training and evaluation trial."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x_batch)  # We don't need the hidden state here
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                logits, _ = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy());
                all_true.extend(y_batch.cpu().numpy())
        val_accuracies.append(accuracy_score(all_true, all_preds))
    return val_accuracies


def find_best_config_for_stage(model_name: str, model_class: nn.Module, model_args: dict,
                               data_args: dict, epochs: int, lr_space: List[float]):
    """Performs a hyperparameter sweep and returns the best results."""
    print(f"\nFinding Best LR for {model_name} on {data_args}")
    best_accuracy = 0
    best_lr = None
    best_curve = None

    train_data, train_labels = generate_parity_data(256 * 20, **data_args)
    val_data, val_labels = generate_parity_data(256 * 5, **data_args)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=256)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_data, val_labels), batch_size=256)

    for lr in lr_space:
        print(f"  Testing LR = {lr}...")
        set_seed(42)
        model = model_class(**model_args)
        learning_curve = run_single_trial(model, train_loader, val_loader, epochs, lr)
        final_accuracy = learning_curve[-1]
        print(f"-> Final Accuracy: {final_accuracy:.2%}")
        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_lr = lr
            best_curve = learning_curve
        plot_hidden_states(model, data_args, model_name, lr=lr)
    return model, best_accuracy, best_lr, best_curve

def plot_hidden_states(model, data_args, title, lr):
    """Generates and plots a t-SNE visualization."""
    model.eval()
    X, y = generate_parity_data(512, **data_args)
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        _, hidden = model(X)
    hidden = hidden.detach().cpu().numpy()
    labels = y.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    hidden_2d = tsne.fit_transform(hidden)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f"{title}\nt-SNE of Final Hidden States", fontsize=16)
    plt.xlabel("t-SNE Dimension 1"); plt.ylabel("t-SNE Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Even', 'Odd'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"plots/{title}/{title}-{data_args['seq_len']}-{str(lr)}.png")
    plt.close()


def run_and_visualize_final_models(lstm_args, hsru_args, data_args, lstm_lr, hsru_lr, epochs):
    """Trains the best models one last time and generates their t-SNE plots."""
    print("\n\n" + "=" * 30 + " FINAL MODEL VISUALIZATION " + "=" * 30)
    # Train and visualize LSTM
    set_seed(42)
    lstm_model = RNNClassifier(**lstm_args)
    _ = run_single_trial(lstm_model, *data_args, epochs, lstm_lr)  # Just to train
    plot_hidden_states(lstm_model, data_args[-1], "LSTM (Best Config)", lstm_lr)
    # Train and visualize HSRnn
    set_seed(42)
    hsru_model = RNNClassifier(**hsru_args)
    _ = run_single_trial(hsru_model, *data_args, epochs, hsru_lr)  # Just to train
    plot_hidden_states(hsru_model, data_args[-1], "HSRnn (Best Config)", hsru_lr)


def run_final_benchmark_with_sweep():
    """Main function to orchestrate the final, fair benchmark."""
    set_seed(42)
    # Configuration
    print("--- Setting up Final Benchmark with LR Sweep for All Models ---")
    INPUT_SIZE = 1
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_CLASSES = 2
    EPOCHS = 25  # Fewer epochs needed for a sweep
    LEARNING_RATE_SPACE = [3e-3, 1e-3, 5e-4, 1e-4]

    # Define Models
    lstm_backbone_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    lstm_classifier_args = {'rnn_backbone': LSTMExtractor(**lstm_backbone_args), 'hidden_size': HIDDEN_SIZE,
                            'output_size': NUM_CLASSES}

    hsru_rnn_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    hsru_classifier_args = {'rnn_backbone': HSRnn(**hsru_rnn_args), 'hidden_size': HIDDEN_SIZE,
                            'output_size': NUM_CLASSES}

    # Define the Curriculum
    curriculum = [
        {"title": "Short, Fixed (SeqLen=10)", "args": {"seq_len": 10, "input_size": 1}},
        {"title": "Medium, Fixed (SeqLen=30)", "args": {"seq_len": 30, "input_size": 1}},
        {"title": "Long, Fixed (SeqLen=60)", "args": {"seq_len": 60, "input_size": 1}},
    ]

    # Run the full benchmark
    lstm_results = []
    hsru_results = []
    final_lstm_curve, final_hsru_curve = None, None
    final_lstm_lr, final_hsru_lr = None, None
    for stage in curriculum:
        # Find the best config for LSTM on this stage
        lstm_modal, lstm_acc, lstm_lr, lstm_curve = find_best_config_for_stage(
            "LSTM", ParityClassifier, lstm_classifier_args, stage['args'], EPOCHS, LEARNING_RATE_SPACE
        )
        lstm_results.append(lstm_acc)

        # Find the best config for HSRnn on this stage
        hsru_modal, hsru_acc, hsru_lr, hsru_curve = find_best_config_for_stage(
            "HSRnn", ParityClassifier, hsru_classifier_args, stage['args'], EPOCHS, LEARNING_RATE_SPACE
        )
        hsru_results.append(hsru_acc)
        final_lstm_curve, final_lstm_lr = lstm_curve, lstm_lr
        final_hsru_curve, final_hsru_lr = hsru_curve, hsru_lr
    # --- 5. Final Report ---
    print("\n\n" + "=" * 35 + " FINAL CURRICULUM BENCHMARK SUMMARY (OPTIMAL CONFIGS) " + "=" * 35)
    stage_labels = [stage['title'] for stage in curriculum]

    print(f"\n| {'Curriculum Stage':<30} | {'LSTM Best Accuracy':<20} | {'HSRnn Best Accuracy':<20} |")
    print(f"|{'-' * 32}|{'-' * 22}|{'-' * 22}|")
    for i, label in enumerate(stage_labels):
        print(f"| {label:<30} | {lstm_results[i]:<20.2%} | {hsru_results[i]:<20.2%} |")
    print("=" * 80)

    # Final Visualization of the Hardest Task
    print("\nVisualizing hidden states for the final, hardest task (SeqLen=60)")

    # Final Report Plot
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, EPOCHS + 1), final_lstm_curve, marker='s', linestyle='--',
             label=f'LSTM (Best LR={final_lstm_lr}) - Final Acc: {lstm_results[-1]:.2%}')
    if final_hsru_curve:
        plt.plot(range(1, EPOCHS + 1), final_hsru_curve, marker='o', linestyle='-',
                 label=f'HSRnn (Best LR={final_hsru_lr}) - Final Acc: {hsru_results[-1]:.2%}')

    plt.title(f"HSRnn vs. LSTM on Final Curriculum Stage ({curriculum[-1]['title']})", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0.45, top=1.02)
    plt.savefig(f"HSRnn vs. LSTM on Final Curriculum Stage ({curriculum[-1]['title']}).png")
    plt.close()

if __name__ == '__main__':
    run_final_benchmark_with_sweep()
