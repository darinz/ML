import numpy as np
from skimage.io import imread

def load_images():
    patch_size = 16
    X_ica = np.zeros((patch_size * patch_size, 40000))
    idx = 0
    for img_num in range(1, 5):
        image = imread(f'images/{img_num}.jpg').astype(float)
        y, x = image.shape
        for i in range(y // patch_size):
            for j in range(x // patch_size):
                patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                X_ica[:, idx] = patch.reshape(-1)
                idx += 1
    X_ica = X_ica[:, :idx]
    W_z = np.linalg.matrix_power((1/X_ica.shape[1]) * X_ica @ X_ica.T, -1//2)
    X_ica = X_ica - np.mean(X_ica, axis=1, keepdims=True)
    X_pca = X_ica.copy()
    X_ica = 2 * W_z @ X_ica
    X_pca = X_pca / np.std(X_pca, axis=1, keepdims=True)
    return X_ica, X_pca 