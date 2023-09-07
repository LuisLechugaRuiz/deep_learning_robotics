## Lesson 2: Activation Functions

### Theory:
Activation functions introduce the necessary non-linearity into the network, allowing it to learn from the error and make adjustments, which is essential for learning complex patterns.

Some of the most common activation functions:

1. **Sigmoid (Logistic) Function**:
    - Formula: \( \sigma(z) = \frac{1}{1+e^{-z}} \)
    - **Range**: (0, 1)
    - **Pros**:
        - Smooth gradient
        - Output values bound between 0 and 1.
    - **Cons**:
        - Can cause vanishing gradient problem.
        - Hard for the network to learn in deeper layers.
        - Not zero-centered.
  
2. **Tanh (Hyperbolic Tangent) Function**:
    - Formula: \( \tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}} \)
    - **Range**: (-1, 1)
    - **Pros**:
        - Zero-centered (can make convergence faster).
    - **Cons**:
        - Can suffer from the vanishing gradient problem in deep networks.

3. **ReLU (Rectified Linear Unit) Function**:
    - Formula: \( \text{ReLU}(z) = \max(0,z) \)
    - **Range**: (0, ∞)
    - **Pros**:
        - Helps mitigate the vanishing gradient problem.
        - Computationally efficient.
    - **Cons**:
        - Neurons can "die" (stop updating during training because their gradient becomes zero).

4. **Leaky ReLU**:
    - Formula: 
        \[
        f(z) =
        \begin{cases} 
        \alpha z & \text{if } z < 0 \\
        z & \text{if } z \geq 0
        \end{cases}
        \]
    - **Pros**:
        - Introduces a small slope to keep the updates alive.

5. **Softmax Function**:
    - **Description**: Often used in the output layer of a classifier to represent probability distributions of multiple classes.
    - Formula: \( \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \) Where \( K \) is the number of classes.
