import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn, optim

from core.hsru import HSRnn
from tests.wrapper import ParityClassifier

device = torch.device("cpu")

def generate_parity_data(n_samples, seq_len, input_size=1):
    """Generates a batch of data for the Temporal Parity Task."""
    X = torch.randint(0, 2, (n_samples, seq_len, input_size)).float()
    y = (X.sum(dim=(1, 2)) % 2 == 1).long()
    return X, y


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


def train_and_eval(model, train_args, title, epochs=200, batch_size=512, lr=1e-4):
    """Performs a full training and evaluation for one stage of the curriculum."""
    print(f"\n--- Training: {title} ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    learning_curve = []
    LEARNING_RATE_SPACE = [3e-3, 1e-3, 5e-4, 1e-4]
    for epoch in range(epochs):
        model.train()
        # Generate fresh data for each epoch
        X, y = generate_parity_data(batch_size, **train_args)
        X, y = X.to(device), y.to(device)

        pred, _ = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation at the end of each epoch
        with torch.no_grad():
            model.eval()
            X_test, y_test = generate_parity_data(512, **train_args)  # Test on a larger set
            X_test, y_test = X_test.to(device), y_test.to(device)
            out, _ = model(X_test)
            acc = accuracy_score(y_test.cpu().numpy(), torch.argmax(out, dim=1).cpu().numpy())
        learning_curve.append(acc)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Val Accuracy: {acc:.2%}")

    print(f"✅ {title}: Final Accuracy = {learning_curve[-1] * 100:.2f}%")
    return learning_curve, model


def plot_hidden_states(model, data_args, title):
    """Generates and plots a t-SNE visualization of the model's final hidden states."""
    model.eval()
    X, y = generate_parity_data(512, **data_args)
    X = X.to(device)
    _, hidden = model(X)
    hidden = hidden.detach().cpu().numpy()
    labels = y.numpy()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
    hidden_2d = tsne.fit_transform(hidden)

    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f"{title}\nt-SNE of Final Hidden States", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Even', 'Odd'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def run_final_benchmark():
    """Main function to orchestrate the curriculum learning benchmark."""
    set_seed(42)
    # Baseline: LSTM (a stronger, more appropriate baseline for this task than GRU)
    lstm_backbone = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
    # Wrapper to make LSTM return only the final hidden state of the last layer
    lstm_extractor = lambda x: lstm_backbone(x)[1][0][-1]
    lstm_model = ParityClassifier(lstm_extractor, hidden_size=64)

    # Your Model: LIFRnn (using the final, corrected version)
    lif_model = ParityClassifier(HSRnn(input_size=1, hidden_size=64, num_layers=2), hidden_size=64)

    # Defining the Curriculum
    curriculum = [
        {"title": "Level 1: Short, Fixed Length (SeqLen=5)", "args": {"seq_len": 5, "input_size": 1}},
        {"title": "Level 1: Short, Fixed Length (SeqLen=10)", "args": {"seq_len": 10, "input_size": 1}},
        {"title": "Level 2: Medium, Fixed Length (SeqLen=60)", "args": {"seq_len": 60, "input_size": 1}},
    ]

    # Running the Curriculum for Both Models
    lstm_final_accs, lif_final_accs = [], []

    print("\n" + "=" * 30 + " BENCHMARKING LSTM BASELINE " + "=" * 30)
    for stage in curriculum:
        learning_curve, lstm_model = train_and_eval(lstm_model, stage["args"], stage["title"], epochs=30, lr=3e-3)
        lstm_final_accs.append(learning_curve[-1])

    print("\n\n" + "=" * 30 + " BENCHMARKING LIFRnn " + "=" * 30)
    for stage in curriculum:
        learning_curve, lif_model = train_and_eval(lif_model, stage["args"], stage["title"], epochs=33, lr=3e-3)
        lif_final_accs.append(learning_curve[-1])

    # Final Report
    print("\n\n" + "=" * 35 + " FINAL CURRICULUM BENCHMARK SUMMARY " + "=" * 35)
    stage_labels = [stage['title'].split(':')[1].strip() for stage in curriculum]

    print(f"\n| {'Curriculum Stage':<25} | {'LSTM Accuracy':<20} | {'LIFRnn Accuracy':<20} |")
    print(f"|{'-' * 27}|{'-' * 22}|{'-' * 22}|")
    for i, label in enumerate(stage_labels):
        print(f"| {label:<25} | {lstm_final_accs[i]:<20.2%} | {lif_final_accs[i]:<20.2%} |")

    # Final Visualization of the Hardest Task
    print("\n--- Visualizing hidden states for the final, hardest task (SeqLen=60) ---")
    plot_hidden_states(lstm_model, curriculum[-1]["args"], "LSTM")
    plot_hidden_states(lif_model, curriculum[-1]["args"], "LIFRnn")


if __name__ == '__main__':
    run_final_benchmark()
