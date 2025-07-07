import numpy as np

# --- Layer Normalization Submodule LN-S ---
def ln_s(z):
    """Layer normalization submodule LN-S: normalize to mean 0, std 1."""
    mu = np.mean(z)
    sigma = np.std(z, ddof=0)  # population std (divide by m)
    return (z - mu) / sigma

# Example for ln_s
z = np.array([1.0, 2.0, 3.0, 4.0])
print("LN-S(z):", ln_s(z))

# --- Layer Normalization LN ---
def ln(z, beta=0.0, gamma=1.0):
    """Layer normalization LN: affine transform after normalization."""
    return beta + gamma * ln_s(z)

# Example for ln
z = np.array([1.0, 2.0, 3.0, 4.0])
beta = 0.5
gamma = 2.0
print("LN(z):", ln(z, beta, gamma))

# --- Matrix Multiplication with Bias ---
def mm(x, W, b):
    """Matrix multiplication with bias: MM_{W, b}(x) = Wx + b."""
    return W @ x + b

# Example for mm
x = np.array([1.0, 2.0])
W = np.array([[1.0, 0.5], [0.5, 1.0]])
b = np.array([0.1, -0.2])
print("MM_{W, b}(x):", mm(x, W, b))

# --- 1D Convolution (Single Channel) ---
def conv1d_s(z, w):
    """1D convolution (valid, zero-padded): filter w over input z."""
    k = len(w)
    l = (k - 1) // 2
    z_padded = np.pad(z, (l, l), mode='constant')
    m = len(z)
    out = np.zeros(m)
    for i in range(m):
        out[i] = np.dot(w, z_padded[i:i+k])
    return out

# Example for conv1d_s
z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
w = np.array([0.2, 0.5, 0.2])
print("Conv1D-S(z):", conv1d_s(z, w))

# --- 1D Convolution (Multiple Channels) ---
def conv1d(z_list, w, C_out):
    """General 1D convolution with multiple input/output channels.
    z_list: list of C input arrays (each length m)
    w: weight tensor of shape (C_out, C_in, k)
    C_out: number of output channels
    Returns: list of C_out output arrays (each length m)
    """
    C_in = len(z_list)
    m = len(z_list[0])
    k = w.shape[2]
    out_list = []
    for i in range(C_out):
        out = np.zeros(m)
        for j in range(C_in):
            out += conv1d_s(z_list[j], w[i, j])
        out_list.append(out)
    return out_list

# Example for conv1d
z_list = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.5, 2.5, 3.5])]
w = np.random.randn(2, 2, 3)  # (C_out=2, C_in=2, k=3)
C_out = 2
out_list = conv1d(z_list, w, C_out)
for i, out in enumerate(out_list):
    print(f"Conv1D output channel {i}: {out}")

# --- 2D Convolution (Single Channel) ---
def conv2d_s(z, w):
    """2D convolution (valid, zero-padded): filter w over input z."""
    k = w.shape[0]
    l = (k - 1) // 2
    z_padded = np.pad(z, ((l, l), (l, l)), mode='constant')
    m = z.shape[0]
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            out[i, j] = np.sum(w * z_padded[i:i+k, j:j+k])
    return out

# Example for conv2d_s
z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
w = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
print("Conv2D-S(z):\n", conv2d_s(z, w))

# --- 2D Convolution (Multiple Channels) ---
def conv2d(z_list, w, C_out):
    """General 2D convolution with multiple input/output channels.
    z_list: list of C input matrices (each m x m)
    w: weight tensor of shape (C_out, C_in, k, k)
    C_out: number of output channels
    Returns: list of C_out output matrices (each m x m)
    """
    C_in = len(z_list)
    m = z_list[0].shape[0]
    k = w.shape[2]
    out_list = []
    for i in range(C_out):
        out = np.zeros((m, m))
        for j in range(C_in):
            out += conv2d_s(z_list[j], w[i, j])
        out_list.append(out)
    return out_list

# Example for conv2d
z_list = [np.random.randn(5, 5), np.random.randn(5, 5)]  # C_in=2
w = np.random.randn(3, 2, 3, 3)  # (C_out=3, C_in=2, k=3, k=3)
C_out = 3
out_list = conv2d(z_list, w, C_out)
for i, out in enumerate(out_list):
    print(f"Conv2D output channel {i}:\n{out}\n") 