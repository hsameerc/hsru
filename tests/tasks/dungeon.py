# test_dungeon_crawler.py

import random

import torch
import torch.nn as nn
import torch.optim as optim

from core.hsru_casual_lm import HSRnnForCausalLM


class DungeonEnv:
    def __init__(self, size=5):
        self.size = size
        self.world = torch.zeros(size, size)  # 0=empty, 1=wall

        # Define fixed locations
        self.start_pos = (0, 0)
        self.key_pos = (size - 2, size - 2)
        self.exit_pos = (size - 1, 0)

        # Simple wall layout
        self.world[1, 0:size - 1] = 1
        self.world[3, 1:size] = 1

        self.has_key = False
        self.player_pos = None
        self.reset()

    def reset(self):
        self.player_pos = self.start_pos
        self.has_key = False

    def step(self, action):  # action is 0:Up, 1:Down, 2:Left, 3:Right
        px, py = self.player_pos
        if action == 0:
            py -= 1  # Up
        elif action == 1:
            py += 1  # Down
        elif action == 2:
            px -= 1  # Left
        elif action == 3:
            px += 1  # Right

        # Check for walls and bounds
        if not (0 <= px < self.size and 0 <= py < self.size and self.world[py, px] == 0):
            return 1  # Hit Wall

        self.player_pos = (px, py)

        if self.player_pos == self.key_pos:
            self.has_key = True
            return 2  # Found Key

        if self.player_pos == self.exit_pos:
            return 3  # On Exit

        return 0  # Moved

    def can_win(self):
        return self.player_pos == self.exit_pos and self.has_key


def generate_dungeon_batch(batch_size, sequence_length, env, device):
    # Input vocab: 0:Moved, 1:HitWall, 2:FoundKey, 3:OnExit
    num_inputs = 4
    inputs = torch.zeros(batch_size, sequence_length, num_inputs, device=device)
    targets = torch.zeros(batch_size, sequence_length, 1, device=device)

    for i in range(batch_size):
        env.reset()
        for t in range(sequence_length):
            action = random.randint(0, 3)
            result = env.step(action)

            inputs[i, t, result] = 1.0
            targets[i, t, 0] = 1.0 if env.can_win() else 0.0

    return inputs, targets


def run_experiment(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(f"Using Device {device}")
    env = DungeonEnv(size=config['dungeon_size'])

    model =HSRnnForCausalLM(
        input_size=4,
        output_size=1,
        hidden_layers_config=config['hidden_layers_config']
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    print("--- Starting Training: Dungeon Crawler Task ---")
    for epoch in range(config['num_epochs']):
        inputs, targets = generate_dungeon_batch(config['batch_size'], config['sequence_length'], env, device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            preds = (torch.sigmoid(outputs) > 0.5).float()
            # Accuracy is tricky here. Let's measure non-zero accuracy.
            # i.e., how often it gets the rare '1' target correct.
            true_pos = ((preds == 1) & (targets == 1)).float().sum()
            total_pos = (targets == 1).float().sum()
            recall = true_pos / (total_pos + 1e-8)
            print(
                f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item():.4f}, Recall on 'Win' state: {recall.item() * 100:.2f}%")

    print("\n--- Starting Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_dungeon_batch(config['batch_size'] * 2, config['sequence_length'], env, device)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()

        accuracy = (preds == targets).all(dim=1).float().mean()
        print(f"Accuracy (entire sequence must be perfect): {accuracy.item() * 100:.2f}%")

        print("\n--- Detailed Example ---")
        input_seq = inputs[0].argmax(dim=-1).tolist()
        target_seq = targets[0].int().squeeze().tolist()
        pred_seq = preds[0].int().squeeze().tolist()

        event_map = {0: "Moved", 1: "Hit Wall", 2: "Found Key", 3: "On Exit"}
        print("Input Event | Target | Predicted | Match?")
        print("------------|--------|-----------|--------")
        for t in range(config['sequence_length']):
            event = event_map[input_seq[t]]
            match = "✔️" if target_seq[t] == pred_seq[t] else "❌"
            print(f"{event:<11} |   {target_seq[t]}    |     {pred_seq[t]}     | {match}")
            if event == "Found Key" or (target_seq[t] == 1 and pred_seq[t] == 1):
                print("------------------------------------------")


if __name__ == "__main__":
    config = {
        'hidden_layers_config': [256],
        'learning_rate': 0.01,
        'batch_size': 256,
        'num_epochs': 1000,
        'sequence_length': 128,
        'dungeon_size': 5,
        'device': 'cuda'
    }

    print("--- The Final Challenge: The Dungeon Crawler ---")
    print("Tests if the model can learn to represent a multi-dimensional state (x, y, has_key).")
    print("This is likely beyond its architectural capabilities. Success is not expected.")
    run_experiment(config)
