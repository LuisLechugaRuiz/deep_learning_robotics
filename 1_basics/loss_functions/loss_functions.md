## Theory:

**Loss functions**, also known as cost functions or objective functions, measure the difference between the predicted output and the actual output (or target). In deep learning, the goal is typically to minimize this difference. The choice of loss function is crucial as it has a direct impact on how the weights in your network are adjusted during training.

Here are some of the most common loss functions:

### 1. Mean Squared Error (MSE)
- Used for regression tasks.
- Formula:\[ L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2 \]
Where \( y \) is the true value, \( \hat{y} \) is the predicted value, and \( N \) is the number of samples.

### 2. Cross-Entropy Loss (Log Loss)
- Used for classification tasks.
- Formula for binary classification:
\[ L(y,p) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)] \]
Where \( y \) is the true label (0 or 1), and \( p \) is the predicted probability of the class being 1.
- For multi-class classification, the formula is extended to handle multiple classes.

### 3. Hinge Loss (max-margin loss)
- Used for Support Vector Machines but can also be used for neural networks in binary classification tasks.
- Formula:
\[ L(y, \hat{y}) = \max(0, 1 - y \times \hat{y}) \]

### 4. Huber Loss
- Used for regression tasks.
- It's a combination of MSE and Mean Absolute Error (MAE) that is less sensitive to outliers than MSE.
- Formula:
\[ L_{\delta}(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
\]

### 5. Categorical Cross-Entropy Loss
- Used for multi-class classification tasks.
- Formula:
\[ L(y,p) = -\sum_{i=1}^{C} y_i \log(p_i) \]
Where \( C \) is the number of classes, \( y \) is a binary indicator (0 or 1) if class label \( c \) is the correct classification for the observation, and \( p \) is the predicted probability that \( y \) is of class \( c \).

### 6. Binary Cross-Entropy Loss
- A special case of cross-entropy loss for binary classification tasks.

### 7. Kullback-Leibler Divergence (KL Divergence)
- Measures how one probability distribution differs from a second, reference probability distribution.
- Often used in Variational Autoencoders (VAEs).
