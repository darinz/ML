# Development Guidelines

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![Linted](https://img.shields.io/badge/linting-flake8%2C%20pylint-green)](https://flake8.pycqa.org/) [![Tested](https://img.shields.io/badge/tests-pytest%20%7C%20unittest-brightgreen)](https://docs.pytest.org/) [![Coverage](https://img.shields.io/badge/coverage-80%25%2B-yellowgreen)](https://coverage.readthedocs.io/) [![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/features/actions)

---

## Table of Contents

- [Code Quality Standards](#code-quality-standards)
- [Machine Learning Specific Guidelines](#machine-learning-specific-guidelines)
- [Debugging Strategies](#debugging-strategies)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Development Environment Setup](#development-environment-setup)
- [Code Review Checklist](#code-review-checklist)

---

### Code Quality Standards

#### Code Style and Documentation
- **PEP 8 Compliance**: Adhere strictly to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code consistency. Use automated linters (e.g., `flake8`, `pylint`) in your workflow.
- **Docstrings**: Every function, class, and module must have a docstring. Use [Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) or [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style. Docstrings should describe parameters, return values, exceptions, and side effects.
- **Type Hints**: All function signatures must include type hints for parameters and return types. This improves code readability and enables static analysis tools.
- **Comments**: Use comments to clarify complex logic, assumptions, and non-obvious decisions. Avoid redundant comments that restate code.
- **Variable Naming**: Use descriptive, unambiguous names. Follow snake_case for variables/functions and PascalCase for classes. Avoid single-letter names except for counters or iterators.
- **Code Organization**: Structure code into modules and packages by logical responsibility. Separate data loading, preprocessing, modeling, and evaluation. Use `__init__.py` to define package interfaces.

#### Testing Best Practices
- **Unit Tests**: Write unit tests for every function and class. Tests should cover normal cases, edge cases, and error conditions. Use `pytest` or `unittest`.
- **Integration Tests**: Test the interaction between components, such as data pipelines and model training workflows. Ensure that the system works as a whole.
- **Test Coverage**: Maintain at least 80% code coverage. Use tools like `coverage.py` to measure and report coverage. Strive for meaningful coverage, not just quantity.
- **Test Data**: Use realistic and diverse datasets for testing. Avoid using production data directly; instead, create representative samples or use synthetic data.
- **Edge Cases**: Explicitly test boundary conditions, invalid inputs, and failure modes. Document any known limitations.
- **Mocking**: Use mocking frameworks (e.g., `unittest.mock`) to isolate tests from external dependencies, such as databases, APIs, or expensive computations.

#### Version Control
- **Commit Messages**: Write concise, descriptive commit messages. Use the imperative mood (e.g., "Add test for data loader"). Reference issues or tickets when relevant.
- **Branch Strategy**: Use feature branches for new development, bugfixes, and experiments. Merge to `main` or `develop` only after review and testing.
- **Code Review**: All changes must be peer-reviewed. Reviewers should check for correctness, style, test coverage, and documentation.
- **Git Hooks**: Set up pre-commit hooks to enforce linting, formatting, and test checks before code is committed. Use tools like `pre-commit` for automation.

### Machine Learning Specific Guidelines

#### Data Handling
- **Data Validation**: Always validate input data for type, shape, range, and missing values. Use assertions or libraries like `pydantic` or `pandera` for schema validation.
- **Data Preprocessing**: Implement preprocessing as reusable, composable pipelines (e.g., using `scikit-learn`'s `Pipeline`). Document each transformation step and its rationale.
- **Feature Engineering**: Clearly document the logic behind feature creation, selection, and transformation. Use version control for feature scripts.
- **Data Versioning**: Use tools like [DVC](https://dvc.org/) or [LakeFS](https://lakefs.io/) to version datasets and track data lineage. Store metadata about data sources and preprocessing steps.
- **Memory Management**: For large datasets, use generators, chunked loading, or out-of-core processing (e.g., `dask`, `vaex`). Profile memory usage and optimize data types.

#### Model Development
- **Reproducibility**: Set random seeds for all libraries (NumPy, PyTorch, TensorFlow, etc.). Log all hyperparameters, environment details, and software versions. Use configuration files (YAML/JSON) for experiment settings.
- **Experimentation**: Track experiments with tools like [MLflow](https://mlflow.org/) or [Weights & Biases](https://wandb.ai/). Log metrics, parameters, artifacts, and code versions for each run.
- **Model Versioning**: Store model artifacts, training scripts, and configuration files in version control. Tag releases and maintain changelogs for model updates.
- **Hyperparameter Tuning**: Use systematic search methods (grid, random, Bayesian) and document the search space and results. Automate tuning with libraries like `Optuna` or `Ray Tune`.
- **Model Evaluation**: Select evaluation metrics appropriate to the problem (e.g., accuracy, F1, ROC-AUC, RMSE). Use stratified cross-validation and report confidence intervals where possible.

#### Performance Optimization
- **Vectorization**: Prefer vectorized operations with NumPy/Pandas over explicit loops for performance and readability. Profile code to identify bottlenecks.
- **Memory Efficiency**: Monitor memory usage with profiling tools. Use efficient data structures (e.g., `float32` instead of `float64` where possible). Release unused memory explicitly.
- **Parallel Processing**: Use Python's `multiprocessing`, `joblib`, or distributed frameworks (e.g., `Dask`, `Ray`) for CPU-bound tasks. Profile and tune parallel workloads.
- **GPU Utilization**: For deep learning, ensure data pipelines keep GPUs busy. Use mixed precision and batch loading. Monitor GPU memory and utilization with `nvidia-smi` or similar tools.
- **Caching**: Cache results of expensive computations (e.g., feature extraction, model predictions) using libraries like `joblib` or `diskcache`.

### Debugging Strategies

#### General Debugging Techniques
1. **Incremental Development**: Build and test code in small, manageable increments. Use version control to checkpoint progress.
2. **Print Debugging**: Insert print statements to trace variable values and execution flow. Remove or convert to logging before production.
3. **Logging**: Use the `logging` module to record events, errors, and warnings. Configure log levels and handlers for different environments.
4. **Interactive Debugging**: Use IPython, Jupyter, or IDE debuggers to inspect variables and step through code interactively.
5. **Visualization**: Plot intermediate results (e.g., data distributions, model outputs) to verify correctness and spot anomalies early.

#### Advanced Debugging Tools
- **Python Debugger (PDB)**: Set breakpoints with `import pdb; pdb.set_trace()`. Use commands to step through code and inspect state.
- **IPython Debugger**: Enable `%pdb` in IPython for automatic post-mortem debugging on exceptions.
- **Memory Profiling**: Use `memory_profiler` or `tracemalloc` to identify memory leaks and optimize usage.
- **Performance Profiling**: Use `cProfile`, `line_profiler`, or `py-spy` to analyze performance bottlenecks. Visualize results with `snakeviz` or `gprof2dot`.
- **Static Analysis**: Run `pylint`, `flake8`, or `mypy` regularly to catch errors, enforce style, and check types.

#### Machine Learning Debugging
- **Gradient Checking**: For custom neural networks, verify gradients numerically to catch implementation errors.
- **Loss Monitoring**: Plot and monitor training/validation loss curves. Watch for divergence, plateaus, or overfitting.
- **Data Inspection**: Visualize samples, distributions, and labels to ensure data integrity. Check for data leakage or label errors.
- **Model Inspection**: Analyze model predictions, feature importances, and error cases. Use tools like SHAP or LIME for interpretability.
- **Overfitting Detection**: Compare training and validation performance. Use regularization, dropout, or early stopping as needed.

### Common Issues and Solutions

#### Python-Specific Issues
- **Type Errors**: Use `type()` and `isinstance()` to check types. Add type hints and run `mypy` to catch issues early.
- **None Values**: Always check for `None` before performing operations. Use `Optional` types and handle missing values explicitly.
- **List Slicing**: Remember that `list[start:end]` includes `start` but excludes `end`. Test edge cases for off-by-one errors.
- **Mutable Defaults**: Avoid mutable default arguments (e.g., `def f(x=[])`). Use `None` and set defaults inside the function.
- **Scope Issues**: Understand variable scope, especially in nested functions, comprehensions, and classes. Avoid shadowing built-ins.

#### NumPy and Scientific Computing
- **Broadcasting**: Check array shapes before operations. Use `np.broadcast_to` or `np.expand_dims` as needed.
- **Shape Assertions**: Assert array shapes after reshaping, stacking, or slicing. Use `assert` statements or custom checks.
- **Data Types**: Choose data types carefully (e.g., `float32` for deep learning, `float64` for precision). Convert types explicitly.
- **Memory Layout**: Consider array memory layout (`C` vs `F` order) for performance. Use `np.ascontiguousarray` if needed.
- **Random Seeds**: Set seeds for all libraries to ensure reproducibility. Document seed values in experiment logs.

#### Machine Learning Specific Issues
- **Data Leakage**: Ensure no information from the test set leaks into training. Split data before preprocessing and feature engineering.
- **Class Imbalance**: Detect and address imbalanced datasets using resampling, class weights, or appropriate metrics.
- **Feature Scaling**: Apply the same scaling/normalization to train and test sets. Fit scalers only on training data.
- **Cross-Validation**: Use stratified or group cross-validation for classification. Avoid data leakage between folds.
- **Model Persistence**: Serialize models with `joblib` or `pickle`. Store version info and dependencies for reproducibility.

### Development Environment Setup

#### Required Tools
- **Python Environment**: Use `venv` or `conda` to manage dependencies. Pin package versions in `requirements.txt` or `environment.yml`.
- **IDE/Editor**: Configure your editor for linting, formatting, and debugging. Recommended: VSCode, PyCharm, or JupyterLab.
- **Version Control**: Set up Git with pre-commit hooks for linting and tests. Use `.gitignore` to exclude sensitive or large files.
- **Testing Framework**: Use `pytest` or `unittest` for automated testing. Integrate with CI/CD pipelines for continuous testing.
- **Code Quality**: Set up pre-commit hooks and CI/CD pipelines to enforce code quality and automate deployments.

#### Recommended Extensions
- **Jupyter Notebooks**: For interactive development, prototyping, and documentation. Use `nbstripout` to clean outputs before committing.
- **Docker**: Containerize environments for reproducibility. Use `Dockerfile` and `docker-compose` for complex setups.
- **MLflow**: Track experiments, models, and metrics. Integrate with CI/CD for automated model deployment.
- **Weights & Biases**: Collaborate on experiment tracking, visualization, and reporting.
- **DVC**: Version control for datasets and models. Integrate with Git for end-to-end reproducibility.

### Code Review Checklist

#### Before Submitting Code
- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have proper docstrings and type hints
- [ ] Unit tests are written and passing
- [ ] No hardcoded values or magic numbers
- [ ] Error handling is implemented appropriately
- [ ] Performance considerations are addressed
- [ ] Documentation is updated

#### For Machine Learning Code
- [ ] Data preprocessing is reproducible
- [ ] Model hyperparameters are documented
- [ ] Evaluation metrics are appropriate
- [ ] Cross-validation is implemented correctly
- [ ] Model artifacts are version controlled
- [ ] Results are reproducible with fixed seeds 