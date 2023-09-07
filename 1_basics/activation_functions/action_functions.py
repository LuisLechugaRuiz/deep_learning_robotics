import torch.nn as nn

# Sigmoid
sigmoid = nn.Sigmoid()

# Tanh
tanh = nn.Tanh()

# ReLU
relu = nn.ReLU()

# Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# Softmax (for classification tasks)
softmax = nn.Softmax(
    dim=1
)  # dim specifies the dimension along which Softmax will be computed
