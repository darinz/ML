[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-%3E=1.18-blue.svg)](https://numpy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-%3E=0.22-orange.svg)](https://scikit-learn.org/stable/)

# Regularization, Model Selection, and Bayesian Methods

This folder contains notes, examples, and code for understanding and applying regularization, model selection, cross-validation, and Bayesian statistics in machine learning. The materials are designed to help you grasp both the theory and practical implementation of these essential concepts.

## Contents

- **01_regularization.md**: Detailed notes on regularization techniques, including L1/L2 regularization, their effects, and practical considerations.
- **02_model_selection.md**: Comprehensive guide to model selection, cross-validation (hold-out, k-fold, leave-one-out), bias-variance tradeoff, and Bayesian statistics. Includes expanded explanations, analogies, and mathematical derivations.
- **regularization_examples.py**: Python code demonstrating regularization in practice, such as ridge and lasso regression.
- **model_selection_and_bayes_examples.py**: Python code for all major equations and calculations from the notes, including:
  - Polynomial model selection
  - Hold-out and k-fold cross-validation
  - Maximum likelihood estimation (MLE)
  - Bayesian linear regression (posterior predictive)
  - MAP estimation (ridge regression)
  - Bayesian logistic regression (MAP)
- **img/**: Folder containing images and diagrams referenced in the notes.

## How to Use

1. **Read the Markdown Notes**
   - Start with `01_regularization.md` for an introduction to regularization.
   - Continue with `02_model_selection.md` for model selection, cross-validation, and Bayesian methods.

2. **Run the Python Examples**
   - Open and run `regularization_examples.py` and `model_selection_and_bayes_examples.py` to see practical implementations of the concepts.
   - Each script is self-contained and prints results for each section.

## Requirements

- Python 3.7+
- numpy
- scikit-learn

You can install the required packages with:

```bash
pip install numpy scikit-learn
```

## Folder Structure

```
06_regularization_model_selection/
├── 01_regularization.md
├── 02_model_selection.md
├── regularization_examples.py
├── model_selection_and_bayes_examples.py
├── img/
└── README.md
```

## Notes
- The code examples are designed for educational purposes and can be adapted for your own experiments.
- The markdown files include both mathematical derivations and intuitive explanations to help you build a strong conceptual foundation.

---

For questions or suggestions, feel free to open an issue or contribute improvements! 