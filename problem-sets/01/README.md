# Problem Sets v1

This directory contains a collection of machine learning problem sets and solutions, organized by topic and assignment. The problems cover a range of foundational topics in machine learning, including linear models, SVMs, kernels, generative models, deep learning, learning theory, unsupervised learning, and reinforcement learning.

## Structure

### Problem Set 1 (ps1/)
**Topics**: Linear Regression, Locally Weighted Linear Regression
- **ps1-problems.md**: Problem descriptions covering linear regression fundamentals
- **q2/**: Locally Weighted Linear Regression (LWLR) implementation
  - `lwlr.py`, `lwlr.m`: LWLR algorithm implementation
  - `plot_lwlr.py`, `plot_lwlr.m`: Visualization scripts
  - `load_data.py`, `load_data.m`: Data loading utilities
  - `data/`: Contains x.dat and y.dat datasets
- **solution/**: Complete solutions and supporting materials
  - `ps1-solution.md`: Comprehensive solution document (38KB)
  - `problem2b_solution.png`: Visualization of LWLR results
  - `q2_solution/`: Complete solution code for question 2

### Problem Set 2 (ps2/)
**Topics**: Classification, Spam Detection, Naive Bayes
- **ps2-problems.md**: Problem descriptions covering classification algorithms
- **q4/**: Spam Classification with Naive Bayes
  - Multiple training datasets: `spam_train_25.arff` to `spam_train_2000.arff`
  - Test dataset: `spam_test.arff`
  - Various dataset sizes for learning curve analysis
- **solution/**: Complete solutions
  - `ps2-solution.md`: Comprehensive solution document (26KB)
  - `q4_solution/`: Complete solution code for spam classification

### Problem Set 3 (ps3/)
**Topics**: Learning Theory, Unsupervised Learning, Clustering
- **ps3-problems.md**: Problem descriptions covering VC dimension, uniform convergence, and clustering
- **q3/**: L1 Regularized Least Squares
  - `l1ls.py`, `l1ls.m`: L1 regularization implementation
  - `load_data.py`, `load_data.m`: Data loading utilities
  - `theta.dat`, `x.dat`, `y.dat`: Training data
- **q4/**: K-Means Clustering
  - `k_means.py`, `k_means.m`: K-means algorithm implementation
  - `draw_clusters.py`, `draw_clusters.m`: Visualization scripts
  - `X.dat`: Clustering dataset
- **solution/**: Complete solutions
  - `ps3-solution.md`: Comprehensive solution document (18KB)
  - `problem4_answer.png`: K-means clustering visualization
  - `q3_solution/`, `q4_solution/`: Complete solution code

### Problem Set 4 (ps4/)
**Topics**: Unsupervised Learning, Dimensionality Reduction, Reinforcement Learning
- **ps4-problems.md**: Problem descriptions covering PCA, ICA, and reinforcement learning
- **q3/**: Principal Component Analysis (PCA) and Independent Component Analysis (ICA)
  - `pca.py`, `pca.m`: PCA implementation
  - `ica.py`, `ica.m`: ICA implementation
  - `plot_pca_filters.py`, `plot_pca_filters.m`: PCA visualization
  - `plot_ica_filters.py`, `plot_ica_filters.m`: ICA visualization
  - `load_images.py`, `load_images.m`: Image loading utilities
  - `images/`: Image datasets for dimensionality reduction
- **q5/**: Reinforcement Learning - Mountain Car Problem
  - `mountain_car.py`, `mountain_car.m`: Mountain car environment
  - `qlearning.py`, `qlearning.m`: Q-learning implementation
  - `plot_mountain_car.py`, `plot_mountain_car.m`: Environment visualization
  - `plot_learning_curves.py`, `plot_learning_curves.m`: Learning curve analysis
- **img/**: Supporting images and diagrams
- **solution/**: Complete solutions
  - `ps4-solution.md`: Comprehensive solution document (24KB)
  - `q3_pca.png`, `q3_ica.png`: PCA and ICA result visualizations
  - `q5_steps_per_episode.png`: Q-learning performance visualization
  - `q3_solution/`, `q5_solution/`: Complete solution code

### Problem Set 5 (ps5/)
**Topics**: Comprehensive Review - All Topics from PS1 to PS4
- **ps5-problems.md**: Problem descriptions covering all foundational machine learning topics
- **Problem 1**: Generalized Linear Models (13 points)
  - Exponential family distributions and log-likelihood concavity
  - Normal distribution case study
- **Problem 2**: Bayesian Linear Regression (15 points)
  - MAP estimation with Gaussian priors
  - Closed-form parameter estimation
- **Problem 3**: Kernels (18 points)
  - Valid kernel proofs and properties
  - Gaussian kernel validation
  - Exponential kernel construction
- **Problem 4**: One-class SVM (18 points)
  - Primal and dual formulations
  - Kernelization capabilities
  - SMO-like optimization algorithms
- **Problem 5**: Uniform Convergence (18 points)
  - Concentration inequalities
  - Sample complexity bounds
  - Union bound applications
- **Problem 6**: Short Answers (40 points)
  - Binary classification with different covariances
  - Perceptron and kernel perceptron analysis
  - Mercer kernel existence
  - Newton's method optimization
  - VC dimension calculations
  - L1-regularized SVM properties
  - Locally weighted regression parameter selection
  - Feature selection strategies

## How to Use

### Getting Started
1. **Choose a Problem Set**: Start with ps1 for linear regression fundamentals, or jump to any topic of interest
2. **Read the Problems**: Open the `ps*-problems.md` file for detailed problem descriptions
3. **Examine the Data**: Each question folder contains relevant datasets and starter code
4. **Check Solutions**: Refer to the `solution/` folders for complete worked solutions

### For Specific Topics

#### Linear Regression (ps1)
- Implement linear regression from scratch
- Learn locally weighted linear regression
- Practice gradient descent optimization

#### Classification (ps2)
- Implement Naive Bayes for spam detection
- Work with ARFF format datasets
- Analyze learning curves with different dataset sizes

#### Learning Theory & Clustering (ps3)
- Study VC dimension and uniform convergence
- Implement L1 regularized least squares
- Practice K-means clustering algorithm

#### Dimensionality Reduction & RL (ps4)
- Implement PCA and ICA for image processing
- Work with the mountain car reinforcement learning environment
- Practice Q-learning algorithm

#### Comprehensive Review (ps5)
- Review all foundational machine learning concepts
- Practice theoretical proofs and mathematical derivations
- Test understanding across multiple topics
- Prepare for comprehensive assessments

### Code Implementation
- **Python Files**: Most implementations are provided in Python (`.py` files)
- **MATLAB Files**: Some problems also include MATLAB implementations (`.m` files)
- **Data Files**: Various formats including `.dat`, `.arff`, and image files
- **Visualization**: Plotting scripts for results analysis

## Requirements

### Core Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Additional Dependencies
- **For ARFF files**: `liac-arff` (optional - solutions show alternative approaches)
- **For image processing**: `PIL` or `opencv-python` (for ps4 image problems)
- **For reinforcement learning**: `gym` (for mountain car environment)

### Installation
```bash
# Install all required packages
pip install numpy pandas scikit-learn matplotlib liac-arff pillow gym
```

## File Formats

### Data Formats
- **`.dat`**: Binary data files (use provided load functions)
- **`.arff`**: Attribute-Relation File Format (spam datasets)
- **`.jpg`**: Image files for dimensionality reduction problems

### Code Formats
- **`.py`**: Python implementations (primary)
- **`.m`**: MATLAB implementations (alternative)
- **`.md`**: Problem descriptions and solutions

## Notes

### Educational Design
- Problems follow classic machine learning course structure
- Solutions include both theoretical explanations and practical implementations
- Visualizations help understand algorithm behavior and results

### Dataset Usage
- **Spam Classification**: Use provided ARFF files or UCI Spambase dataset
- **Clustering**: Synthetic datasets for K-means practice
- **Dimensionality Reduction**: Image datasets for PCA/ICA analysis
- **Reinforcement Learning**: Mountain car environment for Q-learning

### Solution Quality
- Complete worked solutions with detailed explanations
- Both Python and MATLAB implementations where applicable
- Visualization scripts for result analysis
- Performance analysis and learning curves

### Troubleshooting
- If ARFF files cause issues, solutions show alternative dataset loading
- All Python code is tested and ready to run
- MATLAB code provided for users preferring that environment
