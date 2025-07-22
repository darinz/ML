# Problem Sets v1

This directory contains a collection of machine learning problem sets and solutions, organized by topic and assignment. The problems cover a range of foundational topics in machine learning, including linear models, SVMs, kernels, generative models, deep learning, learning theory, unsupervised learning, and reinforcement learning.

## Structure
- **ps1, ps2, ...**: Each subfolder contains a problem set, with a Markdown file describing the problems and (optionally) a `solution/` folder with solutions and supporting code.
- **ps3/**: Problem Set 3, covering learning theory (e.g., VC dimension, uniform convergence, model selection) and unsupervised learning (e.g., clustering, EM, PCA, ICA).
- **ps4/**: Problem Set 4, covering unsupervised learning (e.g., EM for supervised learning, factor analysis, PCA, ICA) and reinforcement learning (e.g., policy iteration, Q-learning, mountain car).
- **q4/**: Contains datasets for the spam classification problem, including multiple training files and a test set in ARFF format.
- **solution/**: Contains worked solutions, including Python scripts for coding problems.

## How to Use
- Read the problem descriptions in the `ps*-problems.md` files (e.g., `ps3/ps3-problems.md`, `ps4/ps4-problems.md`).
- For coding problems, refer to the `solution/` subfolders for example solutions in Python.
- For the spam classification problem (Problem 4 in ps2), you can:
  - Use the provided ARFF datasets in the `q4/` folder (see the solution for how to load and use them in Python).
  - Alternatively, use the UCI Spambase dataset as shown in the provided Python solution.
- Most code solutions are provided as Python scripts and can be run directly. Some problems may be suitable for Jupyter notebooks as well.

## Requirements
- Python 3.x
- Common packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- For ARFF files: `liac-arff` (if you want to load ARFF files directly)

## Getting Started
1. Install the required Python packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib liac-arff
   ```
2. Navigate to the relevant problem set and open the solution script or notebook.
3. Follow the instructions in the problem set and solution files.

## Notes
- The problem sets are designed for educational purposes and follow the structure of classic machine learning courses.
- Datasets in the `q4/` folder are for the spam classification problem and can be used with the provided Python code or your own implementation.
- If you encounter issues with ARFF files, you can use the UCI Spambase dataset directly as shown in the solution.
