import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layer
        # - in_channels = depth of the input -> 3 for RGB.
        # - out_channels = number of filters (kernels).
        # - kernel_size = pixels of the kernel, 1 dim = square, 2 dims = different height and width.
        # - stride = step size (number of pixels) to move the filter.
        # - padding = pixels added to the sides of the input (1 to maintain same spatial dimension)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # (H x W x K) = (32 x 32 x 16)
        # Activation function 1
        self.relu1 = nn.ReLU()  # (0, ∞)
        # Max pooling to reduce spatial dimension of feature map.
        # - kernel_size = same than Convolutional Layer, pixels of the kernel used for the pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # (H / 2 x W / 2 x K) = (16 x 16 x 16)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # (16 x 16 x 32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # (8 x 8 x 32)

        # Fully connected layer
        self.fc = nn.Linear(in_features=8 * 8 * 32, out_features=10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(
            x.size(0), -1
        )  # Here view reshapes the tensor from shape (batch_size, height, width, channels) to (batch_size, height * width * channels)
        x = self.fc(x)
        return x


def get_dataset(batch_size):
    # Define the transformations - Convert images to tensor and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load CIFAR10 training data
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # Split data into training and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, validation_subset = random_split(
        train_dataset, [train_size, val_size]
    )

    # Load CIFAR10 testing data
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, validation_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    learning_rate = 1e-3
    epochs = 10
    batch_size = 64  # Typical values are 32, 64, 128, ...

    # Create the model, criterion, and optimizer
    model = SimpleCNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    train_loader, validation_loader, test_loader = get_dataset(batch_size)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            # Forward pass
            output = model(X_train)
            loss = criterion(output, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        model.eval()  # Set the model to evaluation mode
        best_val_loss = float("inf")
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for X_val, y_val in validation_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                # Only forward pass
                output = model(X_val)
                loss = criterion(output, y_val)
                total_val_loss += loss.item()
                # Track correct predictions
                _, predicted = torch.max(output.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        val_loss = total_val_loss / len(validation_loader)

        # Save best model
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        print(f"Validation accuracy {accuracy}%")


if __name__ == "__main__":
    main()
