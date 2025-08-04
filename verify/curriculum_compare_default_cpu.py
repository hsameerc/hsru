import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn, optim

from src.core.hsru import HSRnn
from tests.wrapper import ParityClassifier
from verify.verify_data import generate_parity_tensors, run_data_diagnostics

device = torch.device("cpu")

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
    for epoch in range(epochs):
        model.train()
        # Generate fresh data for each epoch
        X, y = generate_parity_tensors(batch_size, **train_args)
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
            X_test, y_test = generate_parity_tensors(512, **train_args)  # Test on a larger set
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


def collect_hidden_states(model, data_args, label):
    """Returns hidden states and labels from a model."""
    model.eval()
    X, y = generate_parity_tensors(512, **data_args)
    X = X.to(device)
    _, hidden = model(X)
    return hidden.detach().cpu().numpy(), y.numpy(), [label] * len(y)


def plot_all_hidden_states(models_dict, data_args):
    """
    Plots t-SNE of hidden states from each model in a single, comparative figure.

    Args:
        models_dict (dict): A dictionary of {model_name: model_instance}.
        data_args (dict): Arguments for data generation (e.g., {'seq_len': 200}).
    """
    print("\n" + "=" * 30 + " GENERATING t-SNE VISUALIZATIONS " + "=" * 30)

    all_hidden_states = {}
    all_labels = {}
    for name, model in models_dict.items():
        hidden, labels, _ = collect_hidden_states(model, data_args, name)
        all_hidden_states[name] = hidden
        all_labels[name] = labels  # Labels should be the same for all, but we collect just in case

    all_hidden_2d = {}
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    print("Running t-SNE for all models...")
    for name, hidden in all_hidden_states.items():
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        hidden_2d = tsne.fit_transform(hidden)
        all_hidden_2d[name] = hidden_2d

        # Update global axis limits
        global_x_min = min(global_x_min, hidden_2d[:, 0].min())
        global_x_max = max(global_x_max, hidden_2d[:, 0].max())
        global_y_min = min(global_y_min, hidden_2d[:, 1].min())
        global_y_max = max(global_y_max, hidden_2d[:, 1].max())

    x_buffer = (global_x_max - global_x_min) * 0.05
    y_buffer = (global_y_max - global_y_min) * 0.05

    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 7.5), squeeze=False)
    axes = axes.flatten()  # Ensure axes is always a flat array

    colors = ['#FFD700', '#FF4C4C']
    class_labels = ['Parity 0 (Even)', 'Parity 1 (Odd)']

    fig.suptitle(f"t-SNE Visualization of Final Hidden States (SeqLen={data_args['seq_len']})", fontsize=18, y=0.98)

    for i, name in enumerate(models_dict.keys()):
        ax = axes[i]
        hidden_2d = all_hidden_2d[name]
        labels = all_labels[name]

        for label_val, color, class_name in zip(np.unique(labels), colors, class_labels):
            idx = labels == label_val
            ax.scatter(
                hidden_2d[idx, 0],
                hidden_2d[idx, 1],
                color=color,
                label=class_name,
                alpha=0.8,
                edgecolors='k',
                linewidths=0.5
            )

        ax.set_title(name, fontsize=14)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2" if i == 0 else "")  # Only label y-axis on the first plot

        # Set consistent axis limits for all subplots
        ax.set_xlim(global_x_min - x_buffer, global_x_max + x_buffer)
        ax.set_ylim(global_y_min - y_buffer, global_y_max + y_buffer)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def run_final_benchmark():
    """Main function to orchestrate the curriculum learning benchmark."""
    set_seed(42)
    run_data_diagnostics(num_samples=100000, seq_len=200)
    # Data Analysis and Verification
    print("\n" + "=" * 35 + " DATA ANALYSIS & VERIFICATION " + "=" * 35)
    # Baseline: LSTM (a stronger, more appropriate baseline for this task than GRU)
    lstm_backbone = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True).to(device)
    # Wrapper to make LSTM return only the final hidden state of the last layer
    lstm_extractor = lambda x: lstm_backbone(x)[1][0][-1]
    lstm_model = ParityClassifier(lstm_extractor, hidden_size=64).to(device)

    # GrU
    gru_backbone = nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True).to(device)
    gru_extractor = lambda x: gru_backbone(x)[0][:, -1, :]
    gru_model = ParityClassifier(gru_extractor, hidden_size=64).to(device)

    # Your Model: LIFRnn (using the final, corrected version)
    lif_model = ParityClassifier(HSRnn(input_size=1, hidden_size=64, num_layers=1), hidden_size=64).to(device)

    # Defining the Curriculum
    curriculum = [
        {"title": "Level 1: Short, Fixed Length (SeqLen=5)", "args": {"seq_len": 5, "input_size": 1}},
        {"title": "Level 1: Short, Fixed Length (SeqLen=10)", "args": {"seq_len": 10, "input_size": 1}},
        {"title": "Level 2: Medium, Fixed Length (SeqLen=60)", "args": {"seq_len": 60, "input_size": 1}},
    ]

    # Running the Curriculum for Both Models
    lstm_final_accs, lif_final_accs, gru_final_accs = [], [], []

    print("\n" + "=" * 30 + " BENCHMARKING LSTM BASELINE " + "=" * 30)
    for stage in curriculum:
        learning_curve, lstm_model = train_and_eval(lstm_model, stage["args"], stage["title"], epochs=30, lr=3e-3)
        lstm_final_accs.append(learning_curve[-1])

    print("\n" + "=" * 30 + " BENCHMARKING GRU BASELINE " + "=" * 30)
    for stage in curriculum:
        learning_curve, gru_model = train_and_eval(gru_model, stage["args"], stage["title"], epochs=30, lr=3e-3)
        gru_final_accs.append(learning_curve[-1])

    print("\n\n" + "=" * 30 + " BENCHMARKING HSRU " + "=" * 30)
    for stage in curriculum:
        learning_curve, lif_model = train_and_eval(lif_model, stage["args"], stage["title"], epochs=33, lr=3e-3)
        lif_final_accs.append(learning_curve[-1])

    # Final Report
    print("\n\n" + "=" * 35 + " FINAL CURRICULUM BENCHMARK SUMMARY " + "=" * 35)
    stage_labels = [stage['title'].split(':')[1].strip() for stage in curriculum]

    print(f"\n| {'Curriculum Stage':<40} | {'LSTM Accuracy':<20} | {'GRU Accuracy':<20} | {'HSRU Accuracy':<20}")
    print(f"|{'-' * 42}|{'-' * 22}|{'-' * 22}|{'-' * 22}|")
    for i, label in enumerate(stage_labels):
        print(f"| {label:<40} | {lstm_final_accs[i]:<20%} | {gru_final_accs[i]:<20%}| {lif_final_accs[i]:<21%} |")

    # Final Visualization of the Hardest Task
    print("\n--- Visualizing hidden states for the final, hardest task (SeqLen=60) ---")
    plot_all_hidden_states({
        "LSTM": lstm_model,
        "GRU": gru_model,
        "HSRU": lif_model
    }, curriculum[-1]["args"])


if __name__ == '__main__':
    run_final_benchmark()
