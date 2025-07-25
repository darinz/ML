import numpy as np
import matplotlib.pyplot as plt

def plot_pca_filters(U, patch_size):
    min_val = np.min(U)
    big_filters = min_val * np.ones(((patch_size+1)*patch_size-1, (patch_size+1)*patch_size-1))
    for i in range(patch_size):
        for j in range(patch_size):
            filt = U[:, i*patch_size + j].reshape((patch_size, patch_size))
            big_filters[i*(patch_size+1):(i*(patch_size+1)+patch_size),
                        j*(patch_size+1):(j*(patch_size+1)+patch_size)] = filt
    plt.imshow(big_filters, cmap='gray')
    plt.axis('square')
    plt.axis('off')
    plt.show() 