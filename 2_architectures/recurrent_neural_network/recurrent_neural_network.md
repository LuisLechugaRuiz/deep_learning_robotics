### Recurrent Neural Networks (RNNs):

#### Basic Concept:

RNNs are a type of neural network designed specifically for sequential data. This includes time series data, sentences, audio files, video frames, etc. The fundamental idea behind RNNs is to make use of sequential information by having loops in them, allowing information to persist from one step in the sequence to the next.

#### Architecture:

- **Cells:** At its core, an RNN has a cell that processes each step of the sequence. This cell passes information to itself across time steps.
- **Hidden State:** The RNN maintains a hidden state vector which gets updated at each time step and carries the memory of the network.
- **Input, Output:** At each step, the RNN takes an input xtxt​ and the previous hidden state ht−1ht−1​, and produces an output ytyt​ and a new hidden state htht​.

#### Challenges:

- **Vanishing & Exploding Gradients:** Due to the nature of backpropagation through time, RNNs can suffer from the vanishing and exploding gradient problems. This can make them hard to train on long sequences.
- **Short-term Memory:** Standard RNNs have difficulty in carrying information across many steps, meaning they can struggle to learn dependencies over long sequences.