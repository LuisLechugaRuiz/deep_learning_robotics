import torch.nn as nn
import torch

from simple_neural_network import SimpleNeuralNetwork

# Initialize model
model = SimpleNeuralNetwork(input_size=3, hidden_size=5, output_size=1)
# Initialize loss function
criterion = nn.MSELoss()
# Initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    output = model(X_train)
    # Loss calculation
    loss = criterion(output, y_train)
    # Reset grads
    optimizer.zero_grad()
    # Calculate gradients
    loss.backward()
    # Update weights applying optimization function using grads.
    optimizer.step()
