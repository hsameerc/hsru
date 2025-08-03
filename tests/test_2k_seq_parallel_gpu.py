import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.core.hsru_parallal import HSRnn
from tests.wrapper import RNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    """Ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_parity_data(num_samples, seq_len, input_size):
    """Generates synthetic temporal parity task data."""
    X = torch.randint(0, 2, (num_samples, seq_len, input_size)).float()
    y = (X.sum(dim=(1, 2)) % 2 == 1).long()
    return X, y

def train_and_evaluate(model, train_loader, val_loader, epochs, lr):
    """Trains HSRnn and evaluates on validation set."""
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

        acc = accuracy_score(all_true, all_preds)
        val_accuracies.append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.2%}")

    return val_accuracies

def run_hsrnn_test():
    """Runs the ultimate HSRnn benchmark."""
    print("🚀 Starting HSRnn Benchmark")
    set_seed(42)

    # Config
    INPUT_SIZE = 1
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    NUM_CLASSES = 2
    SEQ_LEN = 2000
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 5e-3
    NUM_SAMPLES = BATCH_SIZE * 50

    print(f"Using device: {device}")
    print("Generating synthetic data...")
    X, y = generate_parity_data(NUM_SAMPLES, SEQ_LEN, INPUT_SIZE)
    dataset = torch.utils.data.TensorDataset(X, y)

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * NUM_SAMPLES)
    val_size = NUM_SAMPLES - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("Initializing HSRnn model...")
    hsru_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    classifier_args = {'rnn_backbone': HSRnn(**hsru_args), 'hidden_size': HIDDEN_SIZE, 'num_classes': NUM_CLASSES}
    model = RNNClassifier(**classifier_args)

    print("Training HSRnn...")
    val_curve = train_and_evaluate(model, train_loader, val_loader, EPOCHS, LR)

    final_acc = val_curve[-1]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n✅ FINAL RESULTS")
    print(f"HSRnn Peak Accuracy: {final_acc:.2%}")
    print(f"Optimal LR: {LR}")
    print(f"Trainable Parameters: {num_params:,}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), val_curve, marker='o', linestyle='-', color='teal')
    plt.title("HSRnn Validation Accuracy on Temporal Parity Task", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0.4, top=1.05)
    plt.tight_layout()
    plt.savefig(f"../plots/test_2k_seq_parallel_gpu/HSRnn-Validation-Accuracy-on-Temporal-Parity-Task.png")

if __name__ == "__main__":
    run_hsrnn_test()