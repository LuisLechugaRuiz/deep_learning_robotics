import torch
import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        # Activation
        self.activation_relu = nn.ReLU()
        self.activation_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_relu(x)
        x = self.layer2(x)
        x = self.activation_relu(x)
        x = self.layer3(x)
        x = self.activation_softmax(x)
        print("Sum of probabilities =", x.sum().item())
        return x


def main():
    simple_nn = SimpleNeuralNetwork(10, 5, 3)
    x = torch.randn(1, 10)
    print("Result = ", simple_nn.forward(x))


if __name__ == "__main__":
    main()
