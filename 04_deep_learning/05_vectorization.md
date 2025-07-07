# 7.5 Vectorization over training examples

As we discussed in Section 7.1, in the implementation of neural networks, we will leverage the parallelism across the multiple examples. This means that we will need to write the forward pass (the evaluation of the outputs) of the neural network and the backward pass (backpropagation) for multiple training examples in matrix notation.

---

**The basic idea.**  The basic idea is simple. Suppose you have a training set with three examples $`x^{(1)}, x^{(2)}, x^{(3)}`$. The first-layer activations for each example are as follows:

```math
\begin{align*}
    z^{[1](1)} &= W^{[1]}x^{(1)} + b^{[1]} \\
    z^{[1](2)} &= W^{[1]}x^{(2)} + b^{[1]} \\
    z^{[1](3)} &= W^{[1]}x^{(3)} + b^{[1]}
\end{align*}
```

Note the difference between square brackets $`[\,]`$, which refer to the layer number, and parenthesis $`(\,)`$, which refer to the training example number. Intuitively, one would implement this using a for loop. It turns out, we can vectorize these operations as well. First, define:

```math
X = \begin{bmatrix} \vert & \vert & \vert \\ x^{(1)} & x^{(2)} & x^{(3)} \\ \vert & \vert & \vert \end{bmatrix} \in \mathbb{R}^{d \times 3}
```

Note that we are stacking training examples in columns and not rows. We can then combine this into a single unified formulation:

```math
Z^{[1]} = \begin{bmatrix} \vert & \vert & \vert \\ z^{[1](1)} & z^{[1](2)} & z^{[1](3)} \\ \vert & \vert & \vert \end{bmatrix} = W^{[1]}X + b^{[1]}
```

You may notice that we are attempting to add $`b^{[1]} \in \mathbb{R}^{d \times 1}`$ to $`W^{[1]}X \in \mathbb{R}^{d \times 3}`$. Strictly following the rules of linear algebra, this is not allowed. In practice however, this addition is performed using **broadcasting**. We create an intermediate $`\tilde{b}^{[1]} \in \mathbb{R}^{d \times 3}`$:

```math
\tilde{b}^{[1]} = \begin{bmatrix} \vert & \vert & \vert \\ b^{[1]} & b^{[1]} & b^{[1]} \\ \vert & \vert & \vert \end{bmatrix}
```

We can then perform the computation: $`Z^{[1]} = W^{[1]}X + \tilde{b}^{[1]}`$. Often times, it is not necessary to explicitly construct $`\tilde{b}^{[1]}`$. By inspecting the dimensions in $`(7.82)`$, you can assume $`\tilde{b}^{[1]} \in \mathbb{R}^{d \times 3}`$ is correctly broadcast to $`W^{[1]}X \in \mathbb{R}^{d \times 3}`$.

The matricization approach as above can easily generalize to multiple layers, with one subtlety though, as discussed below.

---

## Complications/Subtlety in the Implementation

All the deep learning packages or implementations put the data points in the rows of a data matrix. (If the data point itself is a matrix or tensor, then the data is often flattened along the zero-th dimension.) However, most of the deep learning literature and notation stack the data points as columns, as above. Be careful to check the convention used in your implementation.

Many deep learning papers use a similar notation to these notes where the data points are treated as column vectors. There is a simple conversion to deal with the mismatch: in the implementation, all the columns become row vectors, row vectors become column vectors, all the matrices are transposed, and the orders of the matrix multiplications are flipped. 

In the example above, using the row major convention, the data matrix is $`X \in \mathbb{R}^{3 \times d}`$, the first layer weight matrix has dimensionality $`d \times m`$ (instead of $`m \times d`$ as in the two layer neural net section), and the bias vector $`b^{[1]} \in \mathbb{R}^{1 \times m}`$. The computation for the hidden activation becomes

```math
Z^{[1]} = X W^{[1]} + b^{[1]} \in \mathbb{R}^{3 \times m}
```

---

## Why Vectorization Matters

Vectorization is the process of rewriting algorithms to replace explicit loops with array operations. In neural networks, this means expressing computations over all training examples as matrix operations, which are highly optimized in libraries like NumPy, PyTorch, and TensorFlow. This leads to:

- **Significant speedups** due to parallelism and hardware acceleration (SIMD, GPUs).
- **Cleaner, more concise code** that is easier to debug and maintain.
- **Mathematical clarity**: the equations closely match the code.

---

## Mathematical Deep Dive

### Forward Pass: From For-Loops to Matrix Multiplication

Suppose you have $`m`$ training examples, each $`d`$-dimensional, stacked as columns in $`X \in \mathbb{R}^{d \times m}`$ (column-major, as in most ML theory). For a single-layer network:

```math
z^{[1](i)} = W^{[1]} x^{(i)} + b^{[1]}, \quad \forall i = 1, \ldots, m
```

Vectorized:

```math
Z^{[1]} = W^{[1]} X + b^{[1]}
```

where $`Z^{[1]} \in \mathbb{R}^{h \times m}`$, $`W^{[1]} \in \mathbb{R}^{h \times d}`$, $`b^{[1]} \in \mathbb{R}^{h \times 1}`$, and $`h`$ is the number of hidden units.

**Broadcasting**: $`b^{[1]}`$ is automatically broadcast across all columns.

### Row-Major Convention (Implementation)

In most deep learning libraries, data is stored as rows: $`X \in \mathbb{R}^{m \times d}`$. The equivalent computation is:

```math
Z^{[1]} = X W^{[1]T} + b^{[1]}
```

where $`W^{[1]} \in \mathbb{R}^{h \times d}`$, $`b^{[1]} \in \mathbb{R}^{h}`$, and $`Z^{[1]} \in \mathbb{R}^{m \times h}`$.

**Conversion**: To switch between conventions, transpose $`X`$ and $`W`$, and adjust the order of multiplication.

---

## Practical Example: For-Loop vs. Vectorized

Below is a Python example comparing a for-loop implementation to a fully vectorized one for a single-layer neural network:

```python
import numpy as np

# 1000 examples, 4 features, 3 output neurons
np.random.seed(0)
X = np.random.rand(1000, 4)
W = np.random.randn(3, 4)
b = np.random.randn(3)

# For-loop (slow)
outputs_loop = np.zeros((1000, 3))
for i in range(1000):
    for j in range(3):
        outputs_loop[i, j] = np.dot(W[j], X[i]) + b[j]

# Vectorized (fast)
outputs_vec = X @ W.T + b  # shape: (1000, 3)

print('Difference:', np.abs(outputs_loop - outputs_vec).max())
```

This code demonstrates that the vectorized version produces the same result as the for-loop, but is much faster and more concise.

---

## Broadcasting and Tiling

Broadcasting allows you to add a vector to each row or column of a matrix without explicit replication. For example, adding $`b \in \mathbb{R}^h`$ to $`Z \in \mathbb{R}^{m \times h}`$:

- **Broadcasting**: $`Z + b`$ automatically adds $`b`$ to each row.
- **Tiling**: Equivalent to `np.tile(b, (m, 1))` in NumPy, but more efficient.

**Example:**

```python
import numpy as np
Z = np.random.randn(5, 3)
b = np.array([1, 2, 3])
print(Z + b)  # Broadcasting
print(np.tile(b, (5, 1)))  # Tiling
```

---

## Deep Networks: Multi-Layer Vectorization

For a deep network with $`L`$ layers, the forward pass is:

```math
A^{[0]} = X \\
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} \\
A^{[l]} = \sigma(Z^{[l]})
```

or, in row-major code:

```math
A^{[0]} = X \\
Z^{[l]} = A^{[l-1]} W^{[l]T} + b^{[l]} \\
A^{[l]} = \sigma(Z^{[l]})
```

---

## Backpropagation: Vectorized Gradients

The backward pass is also vectorized. For example, the gradient of the loss $`J`$ with respect to $`W^{[l]}`$ is:

```math
\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} (A^{[l-1]})^T
```

where $`\delta^{[l]}`$ is the error term for layer $`l`$.

**In code:**

```python
# Assume dZ is (m, h), A_prev is (m, d)
dW = A_prev.T @ dZ  # (d, h)
db = np.sum(dZ, axis=0)  # (h,)
```

---

## Subtleties and Pitfalls

- **Shape mismatches**: Always check the shapes of your matrices and vectors, especially when switching between row-major and column-major conventions.
- **Batch dimension**: In practice, computations are often performed on mini-batches, not the entire dataset.
- **Numerical stability**: Vectorized code can sometimes hide numerical issues (e.g., overflow in softmax).

---

## Summary Table: Column-Major vs. Row-Major

| Notation         | Theory (Column-Major)         | Implementation (Row-Major) |
|------------------|-------------------------------|----------------------------|
| Data matrix      | $`X \in \mathbb{R}^{d \times m}`$ | $`X \in \mathbb{R}^{m \times d}`$ |
| Weight matrix    | $`W \in \mathbb{R}^{h \times d}`$ | $`W \in \mathbb{R}^{h \times d}`$ |
| Forward formula  | $`Z = W X + b`$                 | $`Z = X W^T + b`$            |
| Output shape     | $`Z \in \mathbb{R}^{h \times m}`$ | $`Z \in \mathbb{R}^{m \times h}`$ |

---

## Worked Example: Two-Layer Neural Network (Vectorized)

Below is a worked example of a two-layer fully-connected neural network using vectorized operations in NumPy:

```python
import numpy as np

# Example: 200 data points, 4 input features, 3 hidden neurons, 1 output
np.random.seed(42)
X = np.random.rand(200, 4)
true_W1 = np.array([[1.2, -0.7, 0.5, 2.0],
                   [0.3, 1.5, -1.0, 0.7],
                   [2.0, 0.1, 0.3, -0.5]])  # (3, 4)
true_b1 = np.array([0.5, -0.2, 0.1])
H = np.maximum(X @ true_W1.T + true_b1, 0)  # Hidden layer (ReLU)
true_W2 = np.array([1.0, -2.0, 0.5])
true_b2 = 0.3
y = H @ true_W2 + true_b2 + np.random.normal(0, 0.2, size=H.shape[0])

# Initialize parameters
W1 = np.random.randn(3, 4)
b1 = np.zeros(3)
W2 = np.random.randn(3)
b2 = 0.0
lr = 0.05

# Training loop
for epoch in range(600):
    # Forward pass
    Z1 = X @ W1.T + b1
    A1 = np.maximum(Z1, 0)  # ReLU
    y_pred = A1 @ W2 + b2
    # Loss (MSE)
    loss = np.mean((y_pred - y) ** 2)
    # Backpropagation
    grad_y_pred = 2 * (y_pred - y) / len(y)
    grad_W2 = A1.T @ grad_y_pred
    grad_b2 = np.sum(grad_y_pred)
    grad_A1 = np.outer(grad_y_pred, W2)
    grad_Z1 = grad_A1 * (Z1 > 0)
    grad_W1 = grad_Z1.T @ X
    grad_b1 = np.sum(grad_Z1, axis=0)
    # Update
    W1 -= lr * grad_W1
    b1 -= lr * grad_b1
    W2 -= lr * grad_W2
    b2 -= lr * grad_b2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Visualize predictions vs. true values
import matplotlib.pyplot as plt
plt.scatter(y, y_pred, alpha=0.6)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Two-Layer Fully-Connected Neural Network')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
```

This example demonstrates how vectorization enables efficient training and prediction for neural networks, and how the code closely matches the mathematical formulation.
