# Self-Supervised Learning: Hands-On Learning Guide

[![Self-Supervised Learning](https://img.shields.io/badge/Self--Supervised%20Learning-Unlabeled%20Data-blue.svg)](https://en.wikipedia.org/wiki/Self-supervised_learning)
[![Foundation Models](https://img.shields.io/badge/Foundation%20Models-Pretraining-green.svg)](https://en.wikipedia.org/wiki/Foundation_model)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Unlabeled Data to Powerful Representations

We've explored the elegant framework of **self-supervised learning**, which addresses the fundamental challenge of learning meaningful representations from unlabeled data. Understanding these concepts is crucial because most real-world data is unlabeled, and manual annotation is expensive, time-consuming, and often impractical.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different self-supervised learning techniques, and develop the intuition needed to build foundation models that can adapt to various downstream tasks.

## From Theoretical Understanding to Hands-On Mastery

We've now explored **large language models** - specialized foundation models for text that leverage the sequential and contextual nature of language. We've seen how language modeling works through the chain rule of probability, how Transformer architectures process text, and how these models can generate coherent text and adapt to new tasks through finetuning, zero-shot learning, and in-context learning.

However, while understanding the theoretical foundations of self-supervised learning and large language models is essential, true mastery comes from **practical implementation**. The concepts we've learned - contrastive learning, language modeling, text generation, and adaptation methods - need to be applied to real problems to develop intuition and practical skills.

This motivates our exploration of **hands-on coding** - the practical implementation of all the self-supervised learning and language model concepts we've learned. We'll put our theoretical knowledge into practice by implementing contrastive learning for computer vision, building language models for text generation, and developing the practical skills needed to create foundation models that can adapt to various downstream tasks.

The transition from theoretical understanding to practical implementation represents the bridge from knowledge to application - taking our understanding of how self-supervised learning and language models work and turning it into practical tools for building powerful AI systems.

In this practical guide, we'll implement complete systems for self-supervised learning and language models, experiment with different techniques, and develop the practical skills needed for real-world applications in computer vision and natural language processing.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master self-supervised learning** through interactive implementations of contrastive learning
2. **Implement foundation models** with pretraining and adaptation strategies
3. **Understand large language models** and their training and deployment
4. **Apply adaptation methods** like linear probing and finetuning
5. **Develop intuition for representation learning** through practical experimentation
6. **Build practical applications** for computer vision and natural language processing

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, arrays)
- Familiarity with machine learning concepts (neural networks, optimization)
- Understanding of deep learning (PyTorch, backpropagation)
- Completion of deep learning and clustering modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 5-6 hours
- **Lesson 2**: 4-5 hours
- **Total**: 10-12 hours

---

## Environment Setup

### Option 1: Using Conda (Recommended)

#### Step 1: Install Miniconda
```bash
# Download Miniconda for your OS
# Windows: https://docs.conda.io/en/latest/miniconda.html
# macOS: https://docs.conda.io/en/latest/miniconda.html
# Linux: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

#### Step 2: Create Environment
```bash
# Navigate to the self-supervised learning directory
cd 09_self-supervised_learning

# Create a new conda environment
conda env create -f code/environment.yaml

# Activate the environment
conda activate self-supervised-learning-lesson

# Verify installation
python -c "import torch, transformers, sklearn; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the self-supervised learning directory
cd 09_self-supervised_learning

# Create virtual environment
python -m venv self-supervised-learning-env

# Activate environment
# On Windows:
self-supervised-learning-env\Scripts\activate
# On macOS/Linux:
source self-supervised-learning-env/bin/activate

# Install requirements
pip install -r code/requirements.txt

# Verify installation
python -c "import torch, transformers, sklearn; print('All packages installed successfully!')"
```

### Option 3: Using Jupyter Notebooks

#### Step 1: Install Jupyter
```bash
# After setting up environment above
pip install jupyter notebook

# Launch Jupyter
jupyter notebook
```

#### Step 2: Create New Notebook
```python
# In a new notebook cell, import required packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
```

---

## Lesson Structure

### Lesson 1: Self-Supervised Learning and Foundation Models (5-6 hours)
**File**: `code/pretraining_examples.py`

#### Learning Goals
- Understand the data labeling problem and motivation for self-supervised learning
- Master supervised pretraining (ImageNet-style classification)
- Implement contrastive learning (SimCLR-style) with data augmentation
- Apply linear probe and finetuning adaptation methods
- Compare different adaptation strategies
- Build practical applications and understand best practices

#### Hands-On Activities

**Activity 1.1: Understanding the Data Labeling Problem**
```python
# Explore why data labeling is expensive and why self-supervised learning is needed
from code.pretraining_examples import demonstrate_labeling_cost

# Demonstrate the high cost of data labeling
demonstrate_labeling_cost()

# Key insight: Manual annotation is prohibitively expensive, motivating self-supervised learning
```

**Activity 1.2: Supervised Pretraining**
```python
# Implement supervised pretraining (ImageNet-style classification)
from code.pretraining_examples import supervised_pretraining_example

# Demonstrate supervised pretraining
model, train_losses, val_accuracies = supervised_pretraining_example()

print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")

# Key insight: Supervised pretraining learns useful representations from labeled data
```

**Activity 1.3: Contrastive Learning Implementation**
```python
# Implement contrastive learning with data augmentation
from code.pretraining_examples import contrastive_learning_example

# Demonstrate contrastive learning
contrastive_model, contrastive_losses = contrastive_learning_example()

print(f"Final contrastive loss: {contrastive_losses[-1]:.4f}")

# Key insight: Contrastive learning learns representations by making similar views close and different views far apart
```

**Activity 1.4: Data Augmentation for Contrastive Learning**
```python
# Explore data augmentation techniques for contrastive learning
from code.pretraining_examples import data_augmentation_example

# Demonstrate data augmentation
data_augmentation_example()

# Key insight: Data augmentation creates different views of the same data for contrastive learning
```

**Activity 1.5: Linear Probe Adaptation**
```python
# Implement linear probe adaptation (feature extraction)
from code.pretraining_examples import linear_probe_example

# Use the pretrained model from Activity 1.2
# Extract features and apply linear probe
features = model.extract_features(test_data)
linear_probe_accuracy = linear_probe_example(model, features)

print(f"Linear probe accuracy: {linear_probe_accuracy:.4f}")

# Key insight: Linear probe evaluates representation quality by training only a linear classifier
```

**Activity 1.6: Finetuning Adaptation**
```python
# Implement finetuning adaptation (full model training)
from code.pretraining_examples import finetuning_example

# Apply finetuning to the pretrained model
finetuned_model, finetune_losses = finetuning_example(model)

print(f"Final finetuning loss: {finetune_losses[-1]:.4f}")

# Key insight: Finetuning adapts all model parameters for the downstream task
```

**Activity 1.7: Comparing Adaptation Methods**
```python
# Compare linear probe vs finetuning performance
from code.pretraining_examples import compare_adaptation_methods

# Compare different adaptation strategies
comparison_results = compare_adaptation_methods()

print("Adaptation methods comparison:")
print(comparison_results)

# Key insight: Different adaptation methods have different trade-offs in performance vs efficiency
```

**Activity 1.8: Feature Visualization and Analysis**
```python
# Visualize learned features and analyze representation quality
from code.pretraining_examples import visualize_features, analyze_feature_quality

# Visualize features using t-SNE
visualize_features(features, labels, "Learned Features Visualization")

# Analyze feature quality
feature_analysis = analyze_feature_quality()

print("Feature quality analysis:")
print(feature_analysis)

# Key insight: Visualization helps understand what the model has learned
```

**Activity 1.9: Practical Considerations**
```python
# Learn practical considerations for self-supervised learning
from code.pretraining_examples import practical_considerations

# Review practical considerations
practical_considerations()

# Key insight: Self-supervised learning requires careful consideration of data, compute, and evaluation
```

#### Experimentation Tasks
1. **Experiment with different data augmentations**: Try different augmentation strategies
2. **Test different contrastive learning temperatures**: Observe how temperature affects learning
3. **Compare different adaptation methods**: Linear probe vs finetuning vs few-shot learning
4. **Analyze feature quality**: Study how different pretraining methods affect downstream performance

#### Check Your Understanding
- [ ] Can you explain why self-supervised learning is important?
- [ ] Do you understand the difference between supervised and contrastive learning?
- [ ] Can you implement contrastive learning from scratch?
- [ ] Do you see the trade-offs between different adaptation methods?

---

### Lesson 2: Large Language Models and Pretraining (4-5 hours)
**File**: `code/pretrain_llm_examples.py`

#### Learning Goals
- Understand language modeling and the chain rule
- Master Transformer input/output interface
- Implement autoregressive text generation with temperature sampling
- Apply finetuning strategies for language models
- Understand zero-shot and in-context learning
- Build practical applications for text generation and understanding

#### Hands-On Activities

**Activity 2.1: Language Modeling and Chain Rule**
```python
# Understand the mathematical foundation of language modeling
from code.pretrain_llm_examples import chain_rule_example

# Demonstrate the chain rule for language modeling
chain_rule_example()

# Key insight: Language modeling decomposes joint probability into conditional probabilities
```

**Activity 2.2: Transformer Input/Output Interface**
```python
# Understand how Transformers process text input and output
from code.pretrain_llm_examples import transformer_io_example

# Demonstrate Transformer input/output interface
transformer_io_example()

# Key insight: Transformers output probability distributions over vocabulary for next token prediction
```

**Activity 2.3: Autoregressive Text Generation**
```python
# Implement text generation with temperature sampling
from code.pretrain_llm_examples import autoregressive_generation_example

# Demonstrate text generation with different temperatures
autoregressive_generation_example()

# Key insight: Temperature controls randomness in text generation
```

**Activity 2.4: Language Model Finetuning**
```python
# Implement finetuning for language models
from code.pretrain_llm_examples import finetuning_example

# Demonstrate language model finetuning
finetuning_example()

# Key insight: Finetuning adapts pretrained language models to specific tasks
```

**Activity 2.5: Zero-Shot and In-Context Learning**
```python
# Explore zero-shot and in-context learning capabilities
from code.pretrain_llm_examples import zero_shot_and_in_context_example

# Demonstrate zero-shot and in-context learning
zero_shot_and_in_context_example()

# Key insight: Large language models can perform new tasks without parameter updates
```

**Activity 2.6: Practical Notes and Best Practices**
```python
# Learn practical considerations for working with language models
from pretrain_llm_examples import practical_notes

# Review practical notes
practical_notes()

# Key insight: Working with language models requires understanding of tokenization, generation, and evaluation
```

#### Experimentation Tasks
1. **Experiment with different temperatures**: Observe how temperature affects text generation
2. **Test different prompts**: See how prompt engineering affects model behavior
3. **Compare different models**: Try different pretrained language models
4. **Analyze generation quality**: Study coherence, diversity, and relevance of generated text

#### Check Your Understanding
- [ ] Can you explain the chain rule for language modeling?
- [ ] Do you understand how Transformers process text?
- [ ] Can you implement text generation with temperature sampling?
- [ ] Do you see the difference between zero-shot and in-context learning?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Contrastive Learning Convergence Problems
```python
# Problem: Contrastive learning doesn't converge or produces poor representations
# Solution: Use proper data augmentation and temperature scaling
def robust_contrastive_learning(model, dataloader, temperature=0.5, epochs=100):
    """Robust contrastive learning with proper augmentation and scaling."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss(temperature=temperature)
    
    # Strong data augmentation for contrastive learning
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            # Create two augmented views
            view1 = transform(data)
            view2 = transform(data)
            
            # Forward pass
            proj1 = model(view1)
            proj2 = model(view2)
            
            # Contrastive loss
            loss = criterion(proj1, proj2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Average loss: {total_loss / len(dataloader):.4f}')
    
    return model
```

#### Issue 2: Language Model Generation Issues
```python
# Problem: Generated text is repetitive or incoherent
# Solution: Use proper sampling strategies and temperature control
def robust_text_generation(model, tokenizer, prompt, max_length=50, temperature=0.7):
    """Robust text generation with proper sampling and temperature control."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generation parameters
    generation_config = {
        'max_length': max_length,
        'temperature': temperature,
        'do_sample': True,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.2,
        'pad_token_id': tokenizer.eos_token_id
    }
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            **generation_config
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
```

#### Issue 3: Memory Issues with Large Models
```python
# Problem: Out of memory when working with large language models
# Solution: Use gradient checkpointing and mixed precision training
def memory_efficient_training(model, dataloader, epochs=10):
    """Memory efficient training with gradient checkpointing and mixed precision."""
    from torch.cuda.amp import GradScaler, autocast
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Mixed precision training
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(dataloader):
            # Mixed precision forward pass
            with autocast():
                outputs = model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}')
    
    return model
```

#### Issue 4: Poor Adaptation Performance
```python
# Problem: Linear probe or finetuning doesn't work well
# Solution: Use proper learning rates and evaluation strategies
def robust_adaptation(pretrained_model, train_data, val_data, method='linear_probe'):
    """Robust adaptation with proper hyperparameter tuning."""
    if method == 'linear_probe':
        # Freeze pretrained model
        for param in pretrained_model.parameters():
            param.requires_grad = False
        
        # Extract features
        features_train = pretrained_model.extract_features(train_data)
        features_val = pretrained_model.extract_features(val_data)
        
        # Train linear classifier
        classifier = LogisticRegression(max_iter=1000, C=1.0)
        classifier.fit(features_train, train_labels)
        
        # Evaluate
        val_accuracy = classifier.score(features_val, val_labels)
        
    elif method == 'finetuning':
        # Unfreeze all parameters
        for param in pretrained_model.parameters():
            param.requires_grad = True
        
        # Use lower learning rate for finetuning
        optimizer = optim.AdamW(pretrained_model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Training loop with early stopping
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            pretrained_model.train()
            train_loss = train_epoch(pretrained_model, train_dataloader, optimizer)
            
            # Validation
            pretrained_model.eval()
            val_accuracy = evaluate(pretrained_model, val_dataloader)
            
            scheduler.step()
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return val_accuracy
```

#### Issue 5: Evaluation Challenges
```python
# Problem: Difficult to evaluate self-supervised learning performance
# Solution: Use multiple evaluation metrics and downstream tasks
def comprehensive_evaluation(pretrained_model, test_datasets):
    """Comprehensive evaluation across multiple downstream tasks."""
    results = {}
    
    for dataset_name, (test_data, test_labels) in test_datasets.items():
        # Extract features
        features = pretrained_model.extract_features(test_data)
        
        # Linear probe evaluation
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(features, test_labels)
        linear_probe_acc = classifier.score(features, test_labels)
        
        # Feature quality metrics
        feature_norm = np.linalg.norm(features, axis=1).mean()
        feature_correlation = np.corrcoef(features.T).mean()
        
        # Clustering evaluation
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(np.unique(test_labels)))
        cluster_labels = kmeans.fit_predict(features)
        
        from sklearn.metrics import adjusted_rand_score
        clustering_score = adjusted_rand_score(test_labels, cluster_labels)
        
        results[dataset_name] = {
            'linear_probe_accuracy': linear_probe_acc,
            'feature_norm': feature_norm,
            'feature_correlation': feature_correlation,
            'clustering_score': clustering_score
        }
    
    return results
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Self-Supervised Learning Level
- [ ] I can explain why self-supervised learning is important
- [ ] I understand the difference between supervised and contrastive learning
- [ ] I can implement contrastive learning from scratch
- [ ] I can apply linear probe and finetuning adaptation

#### Foundation Models Level
- [ ] I can explain the concept of foundation models
- [ ] I understand different adaptation methods
- [ ] I can evaluate representation quality
- [ ] I can apply models to downstream tasks

#### Large Language Models Level
- [ ] I can explain the chain rule for language modeling
- [ ] I understand how Transformers process text
- [ ] I can implement text generation with temperature sampling
- [ ] I can apply zero-shot and in-context learning

#### Practical Application Level
- [ ] I can apply self-supervised learning to real datasets
- [ ] I can work with large language models effectively
- [ ] I can evaluate model performance appropriately
- [ ] I can choose appropriate techniques for different problems

### Progress Tracking

#### Week 1: Self-Supervised Learning and Foundation Models
- **Goal**: Complete Lesson 1
- **Deliverable**: Working contrastive learning implementation with adaptation methods
- **Assessment**: Can you implement contrastive learning and adapt models to downstream tasks?

#### Week 2: Large Language Models and Pretraining
- **Goal**: Complete Lesson 2
- **Deliverable**: Language model implementation with generation and adaptation capabilities
- **Assessment**: Can you work with language models and apply them to text tasks?

---

## Extension Projects

### Project 1: Advanced Self-Supervised Learning Framework
**Goal**: Build a comprehensive self-supervised learning system

**Tasks**:
1. Implement multiple self-supervised learning methods (SimCLR, BYOL, MoCo)
2. Add multi-modal self-supervised learning (vision-language)
3. Create automated evaluation pipelines
4. Build distributed training capabilities
5. Add model compression and deployment tools

**Skills Developed**:
- Advanced self-supervised learning techniques
- Multi-modal learning
- Distributed computing
- Model deployment and optimization

### Project 2: Large Language Model Development
**Goal**: Build a complete language model training and deployment system

**Tasks**:
1. Implement transformer architecture from scratch
2. Add efficient training techniques (gradient checkpointing, mixed precision)
3. Create prompt engineering and evaluation frameworks
4. Build model serving and inference systems
5. Add safety and alignment techniques

**Skills Developed**:
- Transformer architecture design
- Large-scale model training
- Prompt engineering
- Model serving and deployment

### Project 3: Foundation Model Applications
**Goal**: Build practical applications using foundation models

**Tasks**:
1. Implement computer vision applications (object detection, segmentation)
2. Add natural language processing applications (summarization, translation)
3. Create multi-modal applications (image captioning, visual question answering)
4. Build evaluation and monitoring systems
5. Add deployment and scaling capabilities

**Skills Developed**:
- Multi-modal applications
- Real-world system design
- Evaluation and monitoring
- Production deployment

---

## Additional Resources

### Books
- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **"Natural Language Processing with Transformers"** by Lewis Tunstall et al.
- **"Self-Supervised Learning: A Survey"** by Jure Leskovec et al.

### Online Courses
- **Stanford CS224N**: Natural Language Processing with Deep Learning
- **Stanford CS231N**: Convolutional Neural Networks for Visual Recognition
- **MIT 6.S191**: Introduction to Deep Learning

### Practice Datasets
- **ImageNet**: Large-scale image classification dataset
- **CIFAR-10/100**: Small-scale image classification datasets
- **GLUE**: Natural language understanding benchmark
- **HuggingFace Datasets**: Large collection of NLP datasets

### Advanced Topics
- **Vision Transformers**: Self-attention for computer vision
- **Contrastive Learning**: Advanced techniques like BYOL, MoCo, SimSiam
- **Prompt Engineering**: Techniques for effective prompting
- **Model Alignment**: Safety and alignment techniques for large models

---

## Conclusion: The Power of Self-Supervised Learning

Congratulations on completing this comprehensive journey through self-supervised learning! We've explored the fundamental techniques for learning meaningful representations from unlabeled data.

### The Complete Picture

**1. Self-Supervised Learning** - We started with understanding the data labeling problem and explored contrastive learning as a solution.

**2. Foundation Models** - We built models that can adapt to various downstream tasks through pretraining and adaptation.

**3. Large Language Models** - We explored language modeling, text generation, and in-context learning capabilities.

**4. Practical Applications** - We applied these concepts to real-world problems in computer vision and natural language processing.

### Key Insights

- **Data Efficiency**: Self-supervised learning enables learning from vast amounts of unlabeled data
- **Representation Learning**: Contrastive learning creates useful representations without labels
- **Adaptation**: Foundation models can adapt to new tasks through various methods
- **Scale**: Large models exhibit emergent capabilities like in-context learning
- **Trade-offs**: Different adaptation methods balance performance, efficiency, and data requirements

### Looking Forward

This self-supervised learning foundation prepares you for advanced topics:
- **Multi-Modal Learning**: Combining vision, language, and other modalities
- **Efficient Training**: Techniques for training large models with limited resources
- **Model Alignment**: Ensuring models behave safely and as intended
- **Deployment**: Scaling models for production use
- **Emergent Capabilities**: Understanding and leveraging unexpected model behaviors

The principles we've learned here - representation learning, adaptation, and scale - will serve you well throughout your machine learning journey.

### Next Steps

1. **Apply self-supervised learning** to your own datasets
2. **Build foundation models** for specific domains
3. **Explore large language models** for text applications
4. **Contribute to open source** self-supervised learning projects
5. **Continue learning** with more advanced techniques and applications

Remember: Self-supervised learning is the key to unlocking the potential of unlabeled data - it's what enables us to build powerful foundation models that can adapt to any task. Keep exploring, building, and applying these concepts to new problems!

---

**Previous: [Large Language Models](02_pretrain_llm.md)** - Learn how to build and deploy large language models for text generation and understanding.

**Next: [Reinforcement Learning](../10_reinforcement_learning/README.md)** - Explore techniques for learning through interaction and feedback.

## Environment Files

### requirements.txt
```
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
nb_conda_kernels>=2.3.0
```

### environment.yaml
```yaml
name: self-supervised-learning-lesson
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=1.9.0
  - torchvision>=0.10.0
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - scikit-learn>=1.0.0
  - pandas>=1.3.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - notebook>=6.4.0
  - pip
  - pip:
    - transformers>=4.20.0
    - ipykernel
    - nb_conda_kernels
```
