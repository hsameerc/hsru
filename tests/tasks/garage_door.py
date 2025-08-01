import random

from sympy.printing.pytorch import torch
from torch import optim, nn

from core.single import SingleCasualHSRnn


def generate_garage_door_batch(batch_size: int, sequence_length: int, device: torch.device):
    """
    Generates a batch of sequences for the smart garage door opener problem.

    Input vector: [open_button_pressed, close_button_pressed, sensor_is_blocked]
    Output state: 0 for Closed, 1 for Open.
    """
    # Inputs: 3 features per timestep
    inputs = torch.zeros(batch_size, sequence_length, 3, device=device)
    # Targets: 1 feature (door state) per timestep
    targets = torch.zeros(batch_size, sequence_length, 1, device=device)

    for i in range(batch_size):
        # Each sequence starts with the door closed
        door_is_open = 0.0
        for t in range(sequence_length):
            event_type = random.choice(['open', 'close', 'sensor', 'nothing'])
            open_pressed, close_pressed, sensor_blocked = 0.0, 0.0, 0.0

            if event_type == 'open':
                open_pressed = 1.0
            elif event_type == 'close':
                close_pressed = 1.0
                # 50% chance the sensor is also blocked during a close command
                if random.random() > 0.5:
                    sensor_blocked = 1.0
            elif event_type == 'sensor':
                sensor_blocked = 1.0
            # 'nothing' case is all zeros

            inputs[i, t, 0] = open_pressed
            inputs[i, t, 1] = close_pressed
            inputs[i, t, 2] = sensor_blocked

            if open_pressed == 1.0:
                door_is_open = 1.0  # Open command always works
            elif close_pressed == 1.0 and sensor_blocked == 0.0:
                door_is_open = 0.0  # Close command only works if sensor is clear
            # In all other cases (including close when sensor is blocked), state remains the same.

            targets[i, t, 0] = door_is_open

    return inputs, targets


# Training Function
def train_garage(model: SingleCasualHSRnn, config: dict):
    """Trains the model on the garage door task."""
    print("Starting Training: Garage Door Task")
    device = config['device']
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config['num_epochs']):
        inputs, targets = generate_garage_door_batch(
            config['batch_size'], config['sequence_length'], device
        )
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item():.5f}")
    print("--- Training Finished ---")


# Evaluation Function
def evaluate_garage(model: SingleCasualHSRnn, config: dict):
    """Evaluates the model's accuracy on the garage door task."""
    print("\n--- Starting Evaluation ---")
    device = config['device']
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_garage_door_batch(
            config['batch_size'], config['sequence_length'], device
        )
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()

        accuracy = (preds == targets).float().mean()
        print(f"Evaluation Accuracy (per-timestep): {accuracy.item() * 100:.2f}%")

        print("\n------ Example ------")
        print("Input: [Open, Close, Sensor] -> Target | Prediction | State   | Match?")
        print("----------------------------------------------------------------------")
        for t in range(config['sequence_length']):
            input_vec = inputs[0, t].int().tolist()
            target_val = targets[0, t].int().item()
            pred_val = preds[0, t].int().item()
            state_str = "Open  " if target_val == 1 else "Closed"
            match_str = "✔️" if target_val == pred_val else "❌"
            if input_vec == [0, 1, 1]:
                print(
                    f"{str(input_vec):<28} ->   {target_val}   |     {pred_val}    | {state_str} | {match_str} <-- SAFETY CHECK!")
            else:
                print(f"{str(input_vec):<28} ->   {target_val}   |     {pred_val}    | {state_str} | {match_str}")
        print("----------------------------------------------------------------------")


if __name__ == "__main__":
    config = {
        'input_size': 3,
        'output_size': 1,
        'hidden_layers_config': [16],
        'learning_rate': 0.005,
        'batch_size': 128,
        'num_epochs': 4000,
        'sequence_length': 20,
        'device': 'cpu'
    }

    print(f"Using device: {config['device']}")
    model = SingleCasualHSRnn(
        input_size=config['input_size'],
        hidden_layers_config=config['hidden_layers_config'],
        output_size=1
    )
    train_garage(model, config)
    evaluate_garage(model, config)
