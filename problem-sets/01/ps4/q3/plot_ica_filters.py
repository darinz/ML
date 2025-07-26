import numpy as np
import matplotlib.pyplot as plt

def plot_ica_filters(W, patch_size, W_z):
    F = W @ W_z
    norms = np.linalg.norm(F, axis=1)
    idxs = np.argsort(norms)
    min_val = np.min(W)
    big_filters = min_val * np.ones(((patch_size+1)*patch_size-1, (patch_size+1)*patch_size-1))
    for i in range(patch_size):
        for j in range(patch_size):
            idx = idxs[i*patch_size + j]
            filt = W[idx, :].reshape((patch_size, patch_size))
            big_filters[i*(patch_size+1):(i*(patch_size+1)+patch_size),
                        j*(patch_size+1):(j*(patch_size+1)+patch_size)] = filt
    plt.imshow(big_filters, cmap='gray')
    plt.axis('square')
    plt.axis('off')
    plt.show() 