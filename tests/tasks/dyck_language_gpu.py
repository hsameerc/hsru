import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.hsru_casual_lm import HSRnnForCausalLM
from src.core.hsru_parallal import HSRnnForCausalLM


# Data Generation (Iterative Method)
def generate_dyck_batch(batch_size, sequence_length, device):
    """
    Generates a batch of parenthesis sequences iteratively to avoid recursion errors.
    Half the sequences are valid, half are invalid.
    The task is to classify the entire sequence as valid (1) or invalid (0).
    """
    # Vocab: 0='(', 1=')'
    inputs = torch.zeros(batch_size, sequence_length, 2, device=device, dtype=torch.float)
    # Target is a single value per sequence
    targets = torch.zeros(batch_size, 1, device=device)

    for i in range(batch_size):
        balance = 0
        seq = []

        # Generate a sequence body
        for _ in range(sequence_length):
            # If balance is 0, we must open. Otherwise, choose randomly.
            if balance == 0 or (balance < sequence_length // 2 and random.random() < 0.6):
                seq.append('(')
                balance += 1
            else:
                seq.append(')')
                balance -= 1

        # Check if the generated sequence is valid as-is
        is_valid = (balance == 0)

        # We want roughly half valid, half invalid
        make_valid = (i % 2 == 0)

        if make_valid and not is_valid:
            # Force the sequence to be valid by closing all open parens
            # This is a bit of a hack, but ensures valid sequences are possible
            balance_fix = 0
            for t in range(sequence_length):
                if seq[t] == '(':
                    balance_fix += 1
                elif balance_fix > 0:
                    balance_fix -= 1
                else:  # Invalid state, e.g. starts with ')'
                    seq[t] = '('
                    balance_fix += 1
            # Close remaining
            for t in range(sequence_length - 1, -1, -1):
                if balance_fix == 0: break
                if seq[t] == '(':
                    seq[t] = ')'
                    balance_fix -= 2  # a ( -> ) is a change of 2 in balance
            is_valid = True

        elif not make_valid and is_valid:
            # Force the sequence to be invalid by flipping a random char
            idx_to_flip = random.randrange(sequence_length)
            seq[idx_to_flip] = ')' if seq[idx_to_flip] == '(' else '('
            is_valid = False

        # Set the target
        targets[i, 0] = 1.0 if is_valid else 0.0

        # One-hot encode the final sequence
        for t, char in enumerate(seq):
            if char == '(':
                inputs[i, t, 0] = 1.0
            else:
                inputs[i, t, 1] = 1.0

    return inputs, targets


#  Training & Evaluation
def train_dyck(model: HSRnnForCausalLM, config: dict):
    print("Starting Training: Dyck Language (Balanced Parentheses)")
    device = config['device']
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config['num_epochs']):
        inputs, targets = generate_dyck_batch(
            config['batch_size'], config['sequence_length'], device
        )
        # We only care about the final output of the sequence
        final_output = model(inputs)[:, -1, :]
        loss = criterion(final_output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 400 == 0:
            preds = (torch.sigmoid(final_output) > 0.5).float()
            accuracy = (preds == targets).float().mean()
            print(
                f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")

    print("Training Finished")


def evaluate_dyck(model: HSRnnForCausalLM, config: dict):
    print("\nStarting Final Evaluation")
    device = config['device']
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_dyck_batch(
            config['batch_size'] * 2, config['sequence_length'], device
        )
        final_output = model(inputs)[:, -1, :]
        predictions = (torch.sigmoid(final_output) > 0.5).float()
        accuracy = (predictions == targets).float().mean()

        print(f"Final evaluation accuracy: {accuracy.item() * 100:.2f}%")
        print("Note: An accuracy around 50% means the model is essentially guessing.")

        print("\nExample Predictions")
        print("Sequence                               | Target | Predicted")
        print("---------------------------------------|--------|-----------")
        for i in range(10):
            seq_indices = inputs[i].argmax(dim=-1).tolist()
            seq_str = "".join(['(' if j == 0 else ')' for j in seq_indices if j in [0, 1]])
            target_str = "Valid  " if targets[i].item() > 0.5 else "Invalid"
            prediction_str = "Valid  " if predictions[i].item() > 0.5 else "Invalid"
            print(f"{seq_str:<38} | {target_str} | {prediction_str}")


if __name__ == "__main__":
    hs_config = {
        'input_size': 2,
        'output_size': 1,
        'hidden_layers_config': [64],
        'learning_rate': 0.001,
        'batch_size': 128,
        'num_epochs': 4000,
        'sequence_length': 64,
        'device': 'cuda'
    }

    print(f"Using device: {hs_config['device']}")
    print(f"Challenge: Classify sequences of length up to {hs_config['sequence_length']}.")
    print("This problem requires a stack-like memory, which the DualStateRNN lacks.")
    print("We expect the model to fail, achieving an accuracy close to random guessing (50%).\n")

    hs_model = HSRnnForCausalLM(
        input_size=hs_config['input_size'],
        output_size=hs_config['output_size'],
        hidden_layers_config=hs_config['hidden_layers_config']
    )

    train_dyck(hs_model, hs_config)
    evaluate_dyck(hs_model, hs_config)
