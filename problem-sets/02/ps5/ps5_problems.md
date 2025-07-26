# Problem Set 5

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

### (b)

Compute the values of all the partial derivatives on the RHS of your results to the previous question. Then use them to compute the values on the LHS.

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

#### (b)

Find the partial derivatives of the output with respect W^(1) and b^(1), namely $`\frac{\partial y}{\partial W^{(1)}}`$ and $`\frac{\partial y}{\partial b^{(1)}}`$.

#### (c)

Now find the partial derivative of the output with respect to the output of the hidden layer z, that is $`\frac{\partial y}{\partial z}`$.

#### (d)

Finally find the partial derivatives of the output with respect to W^(0) and b^(0), that is $`\frac{\partial y}{\partial W^{(0)}}`$ and $`\frac{\partial y}{\partial b^{(0)}}`$.

### 2.2. Weight initialization

#### (a)

Find the values of $`z`$ (hidden layer output) and $`y`$ (final output) after a forward pass, given that all weights and biases are initialized to 0.

#### (b)

Find the values of the gradients $`\frac{\partial y}{\partial W^{(1)}}`$, $`\frac{\partial y}{\partial b^{(1)}}`$, $`\frac{\partial y}{\partial W^{(0)}}`$, and $`\frac{\partial y}{\partial b^{(0)}}`$.

#### (c)

Observe the values of $`z_i`$, $`\frac{\partial y}{\partial W_i^{(1)}}`$, and $`\frac{\partial y}{\partial b_i^{(1)}}`$, and discuss what this implies for the network's expressiveness.

## 3. The Chain Rule (Optional)

### (a)

Let $`f: \mathbb{R}^n \to \mathbb{R}^m, g: \mathbb{R}^\ell \to \mathbb{R}^n`$. Write the Jacobian of $`f \circ g`$ as a matrix in terms of the Jacobian matrix $`\frac{\partial f}{\partial y}`$ of $`f`$ and the Jacobian matrix $`\frac{\partial g}{\partial x}`$ of $`g`$. Make sure the matrix dimensions line up. What conditions must hold in order for this formula to make sense?

### (b)

Let $`f: \mathbb{R}^n \to \mathbb{R}`$ and $`g: \mathbb{R}^\ell \to \mathbb{R}^n`$. Write the derivative of $`f \circ g`$ as a summation between the partial derivatives $`\frac{\partial f}{\partial y_i}`$ of $`f`$ and the partial derivatives $`\frac{\partial g_i}{\partial x}`$ of $`g`$.

### (c)

What if instead the input of $`g`$ is a matrix $`W \in \mathbb{R}^{p \times q}`$? Can we still represent the derivative $`\frac{\partial g}{\partial W}`$ of $`g`$ as a matrix?
