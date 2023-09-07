import numpy as np
import torch.nn as nn


def mean_squared_error(y, y_pred):
    # nn.MSELoss() in PyTorch
    return np.mean(np.square(y - y_pred))


def cross_entropy_loss_binary_class(y, p):
    # nn.Sigmoid() and nn.BCELoss() in PyTorch -> Use nn.BCEWithLogitsLoss() instead, as it is numerically stable.
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def hinge_loss(y, y_pred):
    # Note: y should be in {-1, 1} for hinge loss, not {0, 1}
    return np.mean(np.max(0, 1 - y * y_pred))


# Also known as Smooth Mean Absolute Error
def huber_loss(y, y_pred, delta):
    # nn.HuberLoss() in PyTorch
    error = y - y_pred
    abs_error = np.abs(error)

    # Quadratic for small errors
    quadratic_mask = abs_error <= delta
    quadratic_loss = 0.5 * np.square(error)

    # Linear for large errors
    linear_loss = delta * abs_error - 0.5 * np.square(delta)

    return np.where(quadratic_mask, quadratic_loss, linear_loss)


def categorical_cross_entropy(y, p):
    # nn.CrossEntropyLoss() in PyTorch - Combines softmax operation and the categorical cross-entropy
    return -np.sum(y * np.log(p))
