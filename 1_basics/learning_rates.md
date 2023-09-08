### Common Learning Rate Schedulers:
1. **StepLR**: Decrease the learning rate by some factor after a specified number of epochs.
2. **ExponentialLR**: Decrease the learning rate exponentially after each epoch.
3. **ReduceLROnPlateau**: Reduce the learning rate when the validation loss plateaus, meaning it stops decreasing.
4. **CosineAnnealingLR**: Adjust the rate according to a cosine schedule.
5. **CyclicLR**: Lets the learning rate cycle between two values with a certain frequency.