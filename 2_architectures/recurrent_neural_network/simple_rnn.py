import torch
import torch.nn as nn


class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNCell, self).__init__()
        self.hidden_size = hidden_size
        # transform input into new representation.
        self.input_weights = nn.Linear(input_size, hidden_size)
        # carry forward the information or "memory" from previous time steps
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size)
        # Hyperbolic tangent as activation function -> range [−1,1]
        # - needs upper bound due to the accumulation of outputs over time.
        # - negative inputs will be mapped strongly negative and the zero inputs will be near zero in the output (better than sigmoid)
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        return self.tanh(self.input_weights(x) + self.recurrent_weights(h_prev))


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.rnn_cell = SimpleRNNCell(input_size, hidden_size)
        # Use a fully connected layer to produce the output based on the final hidden state of the rnn cell.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initializing h to [batch_size, hidden_size].
        h = torch.zeros(x.size(0), self.rnn_cell.hidden_size).to(x.device)
        for t in range(x.size(1)):
            # - batch_size: number of samples being processed simultaneously.
            # - sequence_length: number of timesteps in each input sequence. i.e: words in a sentence.
            # - input_size: dimensionality of the data at each timestep. i.e: dimensionality of the embedding vector.
            # [:, t, :] = [batch_size, input_size] at a timestep t.
            # i.e: processing a word of each sample using his full embedding vector
            h = self.rnn_cell(x[:, t, :], h)
        return self.fc(h)
