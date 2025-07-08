# Dimensionality Reduction

Dimensionality reduction is a fundamental concept in machine learning and data analysis. It refers to techniques that reduce the number of variables (features) in a dataset while preserving as much relevant information as possible. This makes data easier to visualize, interpret, and use for downstream tasks such as classification, clustering, or regression. Dimensionality reduction can also help remove noise and redundancy, speed up algorithms, and reveal hidden structure in the data.

This folder covers two of the most important linear dimensionality reduction techniques:

- **Principal Components Analysis (PCA):** Finds new axes (principal components) that maximize variance and decorrelate the data. Useful for visualization, compression, and noise reduction.
- **Independent Components Analysis (ICA):** Goes beyond PCA by finding statistically independent components in the data. Especially useful for separating mixed signals (e.g., the cocktail party problem).

## Table of Contents

- [Principal Components Analysis (PCA)](01_pca.md)
  - [pca_examples.py](pca_examples.py): Python code for PCA
- [Independent Components Analysis (ICA)](02_ica.md)
  - [ica_examples.py](ica_examples.py): Python code for ICA
- [img/](img/): Figures and diagrams used in the notes

## How to Run the Example Code

1. **Install requirements:**
   - Python 3.7+
   - numpy
   - matplotlib
   - scikit-learn
   
   You can install the requirements with:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

2. **Run the example scripts:**
   - For PCA:
     ```bash
     python pca_examples.py
     ```
   - For ICA:
     ```bash
     python ica_examples.py
     ```

Each script is organized into sections and can be run as a standalone demonstration. The scripts will generate plots to help you visualize the results of dimensionality reduction.

## Further Reading

- [scikit-learn: PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn: FastICA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
- [Wikipedia: Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Wikipedia: Independent component analysis](https://en.wikipedia.org/wiki/Independent_component_analysis)

For detailed mathematical derivations and step-by-step explanations, see the markdown files in this folder. 