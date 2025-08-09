# Generalized Linear Models (GLMs)

[![GLM](https://img.shields.io/badge/GLM-Generalized%20Linear%20Models-blue.svg)](https://en.wikipedia.org/wiki/Generalized_linear_model)
[![Link Functions](https://img.shields.io/badge/Link%20Functions-Transformation-green.svg)](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)
[![Statistics](https://img.shields.io/badge/Statistics-Exponential%20Family-purple.svg)](https://en.wikipedia.org/wiki/Exponential_family)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)

## Overview

GLMs unify linear regression and logistic regression into a single framework using exponential family distributions. They handle various response types (continuous, binary, count, positive) through systematic model construction.

## Materials

### Core Theory
- **[01_exponential_family.md](01_exponential_family.md)** - Mathematical foundation: $p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))$
- **[02_constructing_glm.md](02_constructing_glm.md)** - Systematic GLM construction and three fundamental assumptions

### Implementation
- **[exponential_family_examples.py](exponential_family_examples.py)** - Interactive examples and visualizations
- **[constructing_glm_examples.py](constructing_glm_examples.py)** - Complete GLM framework with real data examples

### Reference Materials
- **exponential_family/** - Academic PDFs from MIT, Princeton, Berkeley, Columbia, Purdue

## Key Concepts

### Exponential Family Form
```math
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))
```

### GLM Construction Recipe
1. **Choose Response Distribution**: Gaussian, Bernoulli, Poisson, Gamma
2. **Define Linear Predictor**: $\eta = \theta^T x$
3. **Specify Link Function**: $\mu = g(\eta)$
4. **Estimate Parameters**: Maximum likelihood

### Three Fundamental Assumptions
1. **Exponential Family Response**: $y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$
2. **Prediction Goal**: $h(x) = \mathbb{E}[y|x]$
3. **Linear Relationship**: $\eta = \theta^T x$

## Common GLM Examples

| Model | Distribution | Link | Use Case |
|-------|-------------|------|----------|
| Linear Regression | Gaussian | Identity | Continuous outcomes |
| Logistic Regression | Bernoulli | Logit | Binary classification |
| Poisson Regression | Poisson | Log | Count data |
| Gamma Regression | Gamma | Inverse | Positive continuous |

## Getting Started

1. Read `01_exponential_family.md` for mathematical foundation
2. Study `02_constructing_glm.md` for systematic approach
3. Run examples in Python files
4. Explore reference materials for deeper understanding

## Prerequisites

- Linear algebra and calculus
- Probability and statistics
- Python programming
- Linear and logistic regression basics

## Applications

- **Healthcare**: Medical diagnosis, survival analysis
- **Finance**: Credit scoring, risk modeling
- **Marketing**: Customer behavior, conversion modeling
- **Ecology**: Species abundance, environmental modeling 