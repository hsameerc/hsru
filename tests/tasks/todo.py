import torch
from torch import optim, nn
from torch.functional import F

from core.hsru_casual_lm import HSRnnForCausalLM


#  Data Generation
def generate_todo_batch(num_tasks: int, batch_size: int, sequence_length: int, device: torch.device):
    """
    Generates a batch of "to-do list" completion sequences.

    Args:
        num_tasks (int): The total number of tasks on the list.
        batch_size (int): The number of sequences in the batch.
        sequence_length (int): The length of each event sequence.
        device: The torch device to place the tensors on.

    Returns:
        A tuple (inputs, targets):
        - inputs: A one-hot tensor of shape (batch_size, sequence_length, num_tasks)
                  representing which task is completed at each timestep.
        - targets: A binary tensor of shape (batch_size, sequence_length, num_tasks)
                   representing the cumulative set of completed tasks at each timestep.
    """
    # Shape: (batch_size, sequence_length)
    completion_indices = torch.randint(0, num_tasks, (batch_size, sequence_length), device=device)

    # Shape: (batch_size, sequence_length, num_tasks)
    input_sequences = F.one_hot(completion_indices, num_classes=num_tasks).float()

    # Shape: (batch_size, sequence_length, num_tasks)
    target_sequences, _ = torch.cummax(input_sequences, dim=1)

    return input_sequences, target_sequences


# Training Function
def train_todo(model: HSRnnForCausalLM, config: dict):
    """Trains the DualStateRNN model on the to-do list task."""
    print("Starting Training: To-Do List Task")
    device = config['device']
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config['num_epochs']):
        inputs, targets = generate_todo_batch(
            config['num_tasks'],
            config['batch_size'],
            config['sequence_length'],
            device
        )

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 250 == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item():.5f}")

    print("Training Finished")


# Evaluation Function
def evaluate_todo(model: HSRnnForCausalLM, config: dict):
    """Evaluates the trained model's accuracy on the to-do list task."""
    print("\nStarting Evaluation")
    device = config['device']
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs, targets = generate_todo_batch(
            config['num_tasks'],
            config['batch_size'],
            config['sequence_length'],
            device
        )

        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()

        # For a strict accuracy, we check if the entire state vector is correct
        # at every single timestep across the whole batch.
        correct_state_vectors = (preds == targets).all(dim=-1).float()
        accuracy = correct_state_vectors.mean()

        print(f"Evaluation Accuracy (exact state match): {accuracy.item() * 100:.2f}%")

        print("\n------ Example ------")
        completion_indices = inputs[0].argmax(dim=-1).tolist()
        print(f"Number of tasks: {config['num_tasks']}")
        print(f"Sequence of completed tasks (by index): {completion_indices}\n")

        print("Timestep | Target State      | Predicted State   | Match?")
        print("---------|-------------------|-------------------|--------")
        for t in range(config['sequence_length']):
            target_str = str(targets[0, t].int().tolist())
            predicted_str = str(preds[0, t].int().tolist())
            match_str = "✔️" if torch.equal(targets[0, t], preds[0, t]) else "❌"
            print(f"t={t:<7} | {target_str:<17} | {predicted_str:<17} | {match_str}")
        print("---------------------")


if __name__ == "__main__":
    NUM_TASKS = 8
    config = {
        'num_tasks': NUM_TASKS,
        'input_size': NUM_TASKS,
        'output_size': NUM_TASKS,
        'hidden_layers_config': [NUM_TASKS * 2],
        'learning_rate': 1e-3,
        'batch_size': 128,
        'num_epochs': 3000,
        'sequence_length': 15,
        'device': 'cpu'
    }
    print(f"Using device: {config['device']}")
    model = HSRnnForCausalLM(
        input_size=config['input_size'],
        output_size=config['output_size'],
        hidden_layers_config=config['hidden_layers_config']
    )
    train_todo(model, config)
    evaluate_todo(model, config)
