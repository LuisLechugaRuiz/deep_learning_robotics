import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet50
resnet = models.resnet50(pretrained=True)

# Remove the final layer (In the case of ResNet, it's fc)
num_features = resnet.fc.in_features

# Replace the final layer with our custom one (e.g., if we have 10 classes)
resnet.fc = nn.Linear(num_features, 10)

# If you want to freeze the layers:
for param in resnet.parameters():
    param.requires_grad = False

# Make sure to not freeze the new layer
for param in resnet.fc.parameters():
    param.requires_grad = True
