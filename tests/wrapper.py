from torch import nn


class RNNClassifier(nn.Module):
    """A generic classifier that wraps a sequential feature extractor."""

    def __init__(self, rnn_backbone, hidden_size, num_classes):
        super().__init__()
        self.rnn = rnn_backbone
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features = self.rnn(x)
        return self.classifier(features)


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


class LSTM_Seq2Seq_Wrapper(nn.Module):
    """A wrapper for nn.LSTM to ensure it returns a full sequence."""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        output_sequence, _ = self.lstm(x)
        return output_sequence


class LSTMExtractor(nn.Module):
    """A wrapper for nn.LSTM to match our RNN interface (returns last hidden state)."""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (final_hidden_state, _) = self.lstm(x)
        return final_hidden_state[-1]
