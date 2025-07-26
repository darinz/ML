# Problem Set 5 Solutions

## 1. Neural Network Chain Rule Warm-Up

Consider the following equations:
*   $`v(a, b, c) = c(a - b)²`$
*   $`a(w, x, y) = (w + x + y)²`$
*   $`b(x, y, z) = (x - y - z)²`$

Below is a network diagram illustrating the relationships between variables:
*   There are four input nodes on the left: $`w`$, $`x`$, $`y`$, $`z`$.
*   There are three intermediate nodes in the middle: $`a`$, $`b`$, $`c`$.
*   There is one output node on the right: $`v`$.
*   Arrows indicate dependencies:
    *   $`w`$, $`x`$, and $`y`$ all point to $`a`$.
    *   $`x`$, $`y`$, and $`z`$ all point to $`b`$.
    *   $`a`$, $`b`$, and $`c`$ all point to $`v`$.

### (a)

Using the multi-variate chain rule (part 1.b), write the derivatives of the output v with respect to each of the input variables: c, w, x, y, z using only partial derivative symbols.

**Solution:**
*   $`\frac{\partial v}{\partial c} = \frac{\partial v}{\partial c}`$
*   $`\frac{\partial v}{\partial w} = \frac{\partial v}{\partial a} \cdot \frac{\partial a}{\partial w}`$
*   $`\frac{\partial v}{\partial x} = \frac{\partial v}{\partial a} \cdot \frac{\partial a}{\partial x} + \frac{\partial v}{\partial b} \cdot \frac{\partial b}{\partial x}`$
*   $`\frac{\partial v}{\partial y} = \frac{\partial v}{\partial a} \cdot \frac{\partial a}{\partial y} + \frac{\partial v}{\partial b} \cdot \frac{\partial b}{\partial y}`$
*   $`\frac{\partial v}{\partial z} = \frac{\partial v}{\partial b} \cdot \frac{\partial b}{\partial z}`$

### (b)

Compute the values of all the partial derivatives on the RHS of your results to the previous question. Then use them to compute the values on the LHS.

**Solution:**
*   $`\frac{\partial v}{\partial a} = 2c(a - b)`$
*   $`\frac{\partial v}{\partial b} = -2c(a - b)`$
*   $`\frac{\partial v}{\partial c} = (a - b)²`$
*   $`\frac{\partial a}{\partial w} = 2(w + x + y)`$
*   $`\frac{\partial a}{\partial x} = 2(w + x + y)`$
*   $`\frac{\partial a}{\partial y} = 2(w + x + y)`$
*   $`\frac{\partial b}{\partial x} = 2(x - y - z)`$
*   $`\frac{\partial b}{\partial y} = -2(x - y - z)`$
*   $`\frac{\partial b}{\partial z} = -2(x - y - z)`$

*   $`\frac{\partial v}{\partial w} = \frac{\partial v}{\partial a} \cdot \frac{\partial a}{\partial w} = 4c(a - b)(w + x + y)`$
*   $`\frac{\partial v}{\partial x} = \frac{\partial v}{\partial a} \cdot \frac{\partial a}{\partial x} + \frac{\partial v}{\partial b} \cdot \frac{\partial b}{\partial x} = 4c(a - b)(w + x + y) - 4c(a - b)(x - y - z) = 4c(a - b)(w + 2y + z)`$
*   $`\frac{\partial v}{\partial y} = \frac{\partial v}{\partial a} \cdot \frac{\partial a}{\partial y} + \frac{\partial v}{\partial b} \cdot \frac{\partial b}{\partial y} = 4c(a - b)(w + x + y) + 4c(a - b)(x - y - z) = 4c(a - b)(w + 2x - z)`$
*   $`\frac{\partial v}{\partial z} = \frac{\partial v}{\partial b} \cdot \frac{\partial b}{\partial z} = 4c(a - b)(x - y - z)`$

## 2. 1-Hidden-Layer Neural Network Gradients and Initialization

### 2.1. Forward and Backward pass

Consider a 1-hidden-layer neural network with a single output unit. Formally the network can be defined by the parameters $`W^{(0)} \in \mathbb{R}^{h \times d}`$, $`b^{(0)} \in \mathbb{R}^h`$; $`W^{(1)} \in \mathbb{R}^{1 \times h}`$ and $`b^{(1)} \in \mathbb{R}`$. The input is given by $`x \in \mathbb{R}^d`$. We will use sigmoid activation for the first hidden layer $`z`$ and no activation for the output $`y`$. Below is a visualization of such a neural network with $`d = 2`$ and $`h = 4`$.

Below the text is a diagram of a neural network:
*   It has an input layer with two nodes, $`x_0`$ and $`x_1`$.
*   A hidden layer with four nodes, $`z_0`$, $`z_1`$, $`z_2`$, and $`z_3`$.
*   An output layer with a single node, $`y`$.
*   There are two bias nodes, represented by dashed circles labeled $`1`$.
*   Connections:
    *   Solid lines connect $`x_0`$ and $`x_1`$ to all $`z`$ nodes. These connections are associated with weights $`W^{(0)}`$.
    *   Dashed lines connect the first bias node (labeled $`1`$, associated with $`b^{(0)}`$) to all $`z`$ nodes.
    *   Solid lines connect all $`z`$ nodes to the $`y`$ node. These connections are associated with weights $`W^{(1)}`$.
    *   Dashed lines connect the second bias node (labeled $`1`$, associated with $`b^{(1)}`$) to the $`y`$ node.

#### (a)

Write out the forward pass for the network using $`x, W^{(0)}, b^{(0)}, z, W^{(1)}, b^{(1)}, \sigma`$ and $`y`$.

**Solution:**
*   $`z = \sigma(W^{(0)}x + b^{(0)})`$
*   $`y = W^{(1)}z + b^{(1)}`$

#### (b)

Find the partial derivatives of the output with respect W^(1) and b^(1), namely $`\frac{\partial y}{\partial W^{(1)}}`$ and $`\frac{\partial y}{\partial b^{(1)}}`$.

**Solution:**
*   $`\frac{\partial y}{\partial W^{(1)}} = z`$
*   $`\frac{\partial y}{\partial b^{(1)}} = 1`$

#### (c)

Now find the partial derivative of the output with respect to the output of the hidden layer z, that is $`\frac{\partial y}{\partial z}`$.

**Solution:**
*   $`\frac{\partial y}{\partial z} = W^{(1)}`$

#### (d)

Finally find the partial derivatives of the output with respect to W^(0) and b^(0), that is $`\frac{\partial y}{\partial W^{(0)}}`$ and $`\frac{\partial y}{\partial b^{(0)}}`$.

**Solution:**
*   **For $`\frac{\partial y}{\partial W^{(0)}}`$:**
    *   $`\frac{\partial z_i}{\partial W_i^{(0)}} = z_i(1 - z_i)x^T \in \mathbb{R}^d`$
    *   $`\frac{\partial y}{\partial W_i^{(0)}} = (\frac{\partial y}{\partial z_i}) \cdot (\frac{\partial z_i}{\partial W_i^{(0)}}) = W_i^{(1)} \cdot z_i(1 - z_i)x^T \in \mathbb{R}^d`$
    *   $`\frac{\partial y}{\partial W^{(0)}} = [W^{(1)} \circ z \circ (1 - z)] x^T \in \mathbb{R}^{h \times d}`$
*   **For $`\frac{\partial y}{\partial b^{(0)}}`$:**
    *   $`\frac{\partial z_i}{\partial b_i^{(0)}} = z_i(1 - z_i) \in \mathbb{R}`$
    *   $`\frac{\partial y}{\partial b_i^{(0)}} = (\frac{\partial y}{\partial z_i}) \cdot (\frac{\partial z_i}{\partial b_i^{(0)}}) = W_i^{(1)} \cdot z_i(1 - z_i) \in \mathbb{R}`$
    *   $`\frac{\partial y}{\partial b^{(0)}} = W^{(1)} \circ z \circ (1 - z) \in \mathbb{R}^h`$

### 2.2. Weight initialization

#### (a)

Find the values of $`z`$ (hidden layer output) and $`y`$ (final output) after a forward pass, given that all weights and biases are initialized to 0.

**Solution:**
```math
z_i = \sigma(W_i^{(0)}x + b_i^{(0)}) = \sigma(0x + 0) = \sigma(0) = \frac{1}{2}
```
```math
y = W^{(1)}z + b^{(1)} = 0 \cdot \frac{1}{2} + 0 = 0
```

#### (b)

Find the values of the gradients $`\frac{\partial y}{\partial W^{(1)}}`$, $`\frac{\partial y}{\partial b^{(1)}}`$, $`\frac{\partial y}{\partial W^{(0)}}`$, and $`\frac{\partial y}{\partial b^{(0)}}`$.

**Solution:**
```math
\frac{\partial y}{\partial W^{(1)}} = z = \frac{1}{2}
```
```math
\frac{\partial y}{\partial b^{(1)}} = 1
```
```math
\frac{\partial y}{\partial W^{(0)}} = [W^{(1)} \circ z \circ (1-z)] x^T = (0 \circ \frac{1}{2} \circ \frac{1}{2}) x^T = 0
```
```math
\frac{\partial y}{\partial b^{(0)}} = W^{(1)} \circ z \circ (1-z) = 0 \circ \frac{1}{2} \circ \frac{1}{2} = 0
```

#### (c)

Observe the values of $`z_i`$, $`\frac{\partial y}{\partial W_i^{(1)}}`$, and $`\frac{\partial y}{\partial b_i^{(1)}}`$, and discuss what this implies for the network's expressiveness.

**Solution:**
If weights are initialized to the same value, all $`z_i`$ will be the same. Similarly, all $`W_i^{(1)}`$ and $`b_i^{(1)}`$ will be the same, effectively reducing the neural network to a single hidden unit. This pattern extends to gradients and subsequent updates during gradient descent, meaning all weights and biases remain the same after any number of steps.

## 3. The Chain Rule (Optional)

### (a)

Let $`f: \mathbb{R}^n \to \mathbb{R}^m, g: \mathbb{R}^\ell \to \mathbb{R}^n`$. Write the Jacobian of $`f \circ g`$ as a matrix in terms of the Jacobian matrix $`\frac{\partial f}{\partial y}`$ of $`f`$ and the Jacobian matrix $`\frac{\partial g}{\partial x}`$ of $`g`$. Make sure the matrix dimensions line up. What conditions must hold in order for this formula to make sense?

**Solution:**
The Chain Rule theorem states that:
```math
\frac{\partial (f \circ g)}{\partial x}(x) = \frac{\partial f}{\partial y}(g(x)) \cdot \frac{\partial g}{\partial x}(x)
```

In order for the dimensions to line up for matrix multiplication, we must have $`\frac{\partial f}{\partial y} \in \mathbb{R}^{m \times n}`$ and $`\frac{\partial g}{\partial x} \in \mathbb{R}^{n \times \ell}`$, since $`f \circ g: \mathbb{R}^\ell \to \mathbb{R}^m`$. Note that by this convention, the gradient of a vector-valued function is:
```math
\frac{\partial f}{\partial y}(y) = \begin{bmatrix}
\frac{\partial f_1}{\partial y_1}(y) & \dots & \frac{\partial f_1}{\partial y_n}(y) \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial y_1}(y) & \dots & \frac{\partial f_m}{\partial y_n}(y)
\end{bmatrix}
```

In order to apply the chain rule, $`f`$ must be differentiable at $`g(x)`$ and $`g`$ must be differentiable at $`x`$.

### (b)

Let $`f: \mathbb{R}^n \to \mathbb{R}`$ and $`g: \mathbb{R}^\ell \to \mathbb{R}^n`$. Write the derivative of $`f \circ g`$ as a summation between the partial derivatives $`\frac{\partial f}{\partial y_i}`$ of $`f`$ and the partial derivatives $`\frac{\partial g_i}{\partial x}`$ of $`g`$.

**Solution:**
```math
\frac{\partial f \circ g}{\partial x} = \sum_{i=1}^n \frac{\partial f}{\partial y_i}(g(x)) \cdot \frac{\partial g_i}{\partial x}(x)
```

### (c)

What if instead the input of $`g`$ is a matrix $`W \in \mathbb{R}^{p \times q}`$? Can we still represent the derivative $`\frac{\partial g}{\partial W}`$ of $`g`$ as a matrix?

**Solution:**
No, we cannot. The derivative of $`g: \mathbb{R}^{p \times q} \to \mathbb{R}^n`$ would be represented as a three-dimensional $`n \times p \times q`$ tensor. In practice, people often **flatten** the input matrix $`W`$ to a vector $`\text{vec}(W) \in \mathbb{R}^{pq}`$. Then we can write the derivative of $`g`$ as a Jacobian matrix, $`\frac{\partial g}{\partial \text{vec}(W)} \in \mathbb{R}^{n \times pq}`$. Then we must remember to un-flatten the derivative later when we update the matrix $`W`$.
