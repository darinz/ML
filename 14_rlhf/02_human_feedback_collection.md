# Human Feedback Collection

This guide provides a comprehensive overview of human feedback collection for reinforcement learning from human feedback (RLHF) systems. We'll explore different types of feedback, collection strategies, annotation guidelines, quality control measures, and practical implementation considerations.

## From Theoretical Foundations to Human Feedback Collection

We've now explored **the fundamentals of reinforcement learning for language models** - the mathematical and conceptual foundations that underpin modern RLHF systems. We've seen how traditional RL concepts are adapted for language generation tasks, the unique challenges this domain presents, and the mathematical frameworks that enable learning from human preferences rather than supervised labels.

However, while understanding the theoretical foundations is crucial, **the quality and quantity of human feedback** is what ultimately determines the success of RLHF systems. Consider training a model to be helpful, harmless, and honest - the effectiveness of this training depends entirely on how well we collect and structure human feedback about what constitutes good behavior.

This motivates our exploration of **human feedback collection** - the systematic process of gathering, structuring, and validating human preferences and judgments about model outputs. We'll see how different types of feedback (binary preferences, rankings, ratings, natural language explanations) capture different aspects of human judgment, how to design effective annotation guidelines and quality control measures, and how to mitigate biases and ensure diverse perspectives in the feedback collection process.

The transition from theoretical foundations to human feedback collection represents the bridge from understanding how RLHF works to implementing the data collection pipeline that makes it possible - taking our knowledge of the mathematical framework and applying it to the practical challenge of gathering high-quality human feedback.

In this section, we'll explore human feedback collection, understanding how to design effective data collection strategies that enable successful RLHF training.

## Table of Contents

- [Overview](#overview)
- [Types of Human Feedback](#types-of-human-feedback)
- [Data Collection Strategies](#data-collection-strategies)
- [Annotation Guidelines](#annotation-guidelines)
- [Quality Control](#quality-control)
- [Bias Mitigation](#bias-mitigation)
- [Implementation Examples](#implementation-examples)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Overview

Human feedback collection is the foundation of RLHF systems. Unlike traditional supervised learning that relies on labeled examples, RLHF learns from human preferences, judgments, and feedback about model outputs. This approach enables training models that align with human values and preferences rather than just mimicking training data.

### Key Principles

**1. Preference-Based Learning**: Learn from relative preferences rather than absolute labels
**2. Subjective Evaluation**: Capture human judgments about quality, safety, and usefulness
**3. Iterative Refinement**: Continuously improve based on feedback
**4. Diverse Perspectives**: Incorporate feedback from various populations and viewpoints

### Mathematical Framework

The goal is to learn a reward function $`R_\phi(x, y)`$ that captures human preferences:

```math
R_\phi(x, y) = f_\phi(\text{encode}(x, y))
```

Where:
- $`x`$: Input prompt/context
- $`y`$: Generated response
- $`f_\phi`$: Neural network with parameters $`\phi`$
- $`\text{encode}`$: Encoder for prompt-response pairs

## Types of Human Feedback

### 1. Binary Preferences

**Definition**: Annotators choose between two responses to the same prompt

**Format**: $(x, y_1, y_2, preference)$ where preference $\in \{1, 2\}$

**Example**:
```
Prompt: "Explain quantum computing in simple terms"
Response A: "Quantum computing uses quantum bits that can be 0, 1, or both at once."
Response B: "Quantum computing is like having a super-fast calculator that uses tiny particles."
Preference: 2 (Response B preferred)
```

**Advantages**:
- Simple and fast annotation
- Clear preference signal
- Easy to implement

**Disadvantages**:
- Limited information (only relative preference)
- May not capture fine-grained differences
- Requires careful response pairing

### 2. Ranking

**Definition**: Annotators order multiple responses by quality

**Format**: $(x, y_1, y_2, \ldots, y_k, ranking)$ where ranking is permutation of $\{1, 2, \ldots, k\}$

**Example**:
```
Prompt: "Write a short story about a robot learning to paint"
Response A: "The robot picked up a brush and painted a beautiful sunset."
Response B: "In a world where robots could feel, one discovered art."
Response C: "The robot's circuits buzzed as it created its first masterpiece."
Ranking: [2, 1, 3] (B > A > C)
```

**Advantages**:
- More informative than binary preferences
- Captures fine-grained differences
- Efficient for multiple comparisons

**Disadvantages**:
- More complex annotation task
- May be inconsistent for similar quality responses
- Requires careful response selection

### 3. Rating

**Definition**: Annotators score responses on Likert scales

**Format**: $(x, y, rating)$ where rating $\in \{1, 2, \ldots, L\}$

**Example**:
```
Prompt: "Summarize the benefits of renewable energy"
Response: "Renewable energy sources like solar and wind provide clean, sustainable power that reduces greenhouse gas emissions and creates jobs."
Rating: 4/5 (Very Good)
```

**Advantages**:
- Absolute quality assessment
- Fine-grained evaluation
- Easy to aggregate

**Disadvantages**:
- Subjective scale interpretation
- May not capture relative preferences well
- Requires calibration across annotators

### 4. Natural Language Feedback

**Definition**: Annotators provide written explanations of their preferences

**Format**: $(x, y_1, y_2, preference, explanation)$

**Example**:
```
Prompt: "Explain machine learning to a 10-year-old"
Response A: "Machine learning is when computers learn from examples, like how you learn to recognize dogs by seeing many pictures of dogs."
Response B: "Machine learning uses algorithms to find patterns in data and make predictions."
Preference: 1 (Response A)
Explanation: "Response A uses a concrete analogy that a child can understand, while Response B uses technical terms that might confuse them."
```

**Advantages**:
- Rich, interpretable feedback
- Captures reasoning behind preferences
- Useful for improving annotation guidelines

**Disadvantages**:
- Time-consuming to collect and process
- Difficult to scale
- Requires qualitative analysis

## Data Collection Strategies

### Active Learning

**Principle**: Select the most informative examples for annotation

**Strategies**:

**1. Uncertainty Sampling**:
```python
def uncertainty_sampling(model, unlabeled_data, batch_size=100):
    """
    Select examples where the model is most uncertain
    
    Args:
        model: Current reward model
        unlabeled_data: Pool of unlabeled examples
        batch_size: Number of examples to select
    
    Returns:
        selected_indices: Indices of selected examples
    """
    uncertainties = []
    
    for i, (x, y1, y2) in enumerate(unlabeled_data):
        # Get model predictions
        r1 = model.predict(x, y1)
        r2 = model.predict(x, y2)
        
        # Compute uncertainty (e.g., difference between predictions)
        uncertainty = abs(r1 - r2)
        uncertainties.append((i, uncertainty))
    
    # Select examples with highest uncertainty
    uncertainties.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in uncertainties[:batch_size]]
    
    return selected_indices
```

**2. Diversity Sampling**:
```python
def diversity_sampling(embeddings, batch_size=100):
    """
    Select diverse examples to maximize coverage
    
    Args:
        embeddings: Embeddings of unlabeled examples
        batch_size: Number of examples to select
    
    Returns:
        selected_indices: Indices of selected examples
    """
    from sklearn.cluster import KMeans
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=batch_size, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Select one example from each cluster
    selected_indices = []
    for cluster_id in range(batch_size):
        cluster_examples = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_examples) > 0:
            # Select example closest to cluster center
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(embeddings[cluster_examples] - center, axis=1)
            closest_idx = cluster_examples[np.argmin(distances)]
            selected_indices.append(closest_idx)
    
    return selected_indices
```

### Diversity Sampling

**Goal**: Ensure coverage of different topics, styles, and difficulty levels

**Implementation**:
```python
class DiversitySampler:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.selected_features = []
    
    def select_diverse_batch(self, candidates, batch_size=100):
        """
        Select diverse batch of examples
        
        Args:
            candidates: List of candidate examples
            batch_size: Number of examples to select
        
        Returns:
            selected: Selected examples
        """
        selected = []
        
        for _ in range(batch_size):
            if not candidates:
                break
            
            # Extract features for remaining candidates
            candidate_features = [self.feature_extractor(c) for c in candidates]
            
            # Find candidate most different from already selected
            max_diversity = -1
            best_candidate = None
            
            for i, features in enumerate(candidate_features):
                diversity = self._compute_diversity(features, self.selected_features)
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = i
            
            if best_candidate is not None:
                selected.append(candidates[best_candidate])
                self.selected_features.append(candidate_features[best_candidate])
                candidates.pop(best_candidate)
        
        return selected
    
    def _compute_diversity(self, features, selected_features):
        """Compute diversity score"""
        if not selected_features:
            return 1.0
        
        # Compute average distance to selected features
        distances = [np.linalg.norm(features - sf) for sf in selected_features]
        return np.mean(distances)
```

### Quality Control

**Multi-Annotator Agreement**:
```python
def compute_agreement(annotations, method='kappa'):
    """
    Compute inter-annotator agreement
    
    Args:
        annotations: List of annotations from multiple annotators
        method: Agreement metric ('kappa', 'fleiss', 'krippendorff')
    
    Returns:
        agreement_score: Agreement score between 0 and 1
    """
    if method == 'kappa':
        return cohen_kappa_score(annotations)
    elif method == 'fleiss':
        return fleiss_kappa(annotations)
    elif method == 'krippendorff':
        return krippendorff_alpha(annotations)
    else:
        raise ValueError(f"Unknown agreement method: {method}")
```

**Consistency Checks**:
```python
def check_annotation_consistency(annotations, threshold=0.7):
    """
    Check annotation consistency and flag problematic examples
    
    Args:
        annotations: List of annotations for each example
        threshold: Minimum agreement threshold
    
    Returns:
        consistent_examples: Examples with high agreement
        inconsistent_examples: Examples with low agreement
    """
    consistent_examples = []
    inconsistent_examples = []
    
    for i, example_annotations in enumerate(annotations):
        agreement = compute_agreement(example_annotations)
        
        if agreement >= threshold:
            consistent_examples.append(i)
        else:
            inconsistent_examples.append(i)
    
    return consistent_examples, inconsistent_examples
```

## Annotation Guidelines

### Clear Instructions

**Example Guidelines**:
```
EVALUATION CRITERIA

1. HELPFULNESS (1-5 scale)
   - Does the response address the user's question or request?
   - Is the information accurate and relevant?
   - Does it provide the level of detail appropriate for the context?

2. HARMLESSNESS (1-5 scale)
   - Is the response safe and appropriate?
   - Does it avoid harmful, offensive, or dangerous content?
   - Is it suitable for a general audience?

3. HONESTY (1-5 scale)
   - Is the response truthful and accurate?
   - Does it acknowledge limitations when uncertain?
   - Does it avoid making false claims?

4. CLARITY (1-5 scale)
   - Is the response clear and well-structured?
   - Is it easy to understand?
   - Does it use appropriate language for the audience?
```

### Example Demonstrations

**High-Quality Annotation Example**:
```
Prompt: "What are the benefits of exercise?"

Response A: "Exercise helps you stay healthy and strong. It makes your heart work better and can help you live longer. You should try to exercise for at least 30 minutes most days."

Response B: "Exercise is good for you."

Annotation: A > B
Reasoning: Response A provides specific benefits and actionable advice, while Response B is too vague and unhelpful.
```

**Low-Quality Annotation Example**:
```
Prompt: "Explain photosynthesis"

Response A: "Photosynthesis is the process where plants convert sunlight into energy."

Response B: "Plants use sunlight to make food through photosynthesis."

Annotation: A = B
Reasoning: Both responses are equally accurate and helpful.
```

### Consistency Checks

**Inter-Annotator Agreement Monitoring**:
```python
class AnnotationQualityMonitor:
    def __init__(self, min_agreement=0.7):
        self.min_agreement = min_agreement
        self.agreement_history = []
    
    def monitor_batch(self, batch_annotations):
        """
        Monitor annotation quality for a batch
        
        Args:
            batch_annotations: Annotations for current batch
        
        Returns:
            quality_report: Report on annotation quality
        """
        agreements = []
        for example_annotations in batch_annotations:
            agreement = compute_agreement(example_annotations)
            agreements.append(agreement)
        
        avg_agreement = np.mean(agreements)
        self.agreement_history.append(avg_agreement)
        
        quality_report = {
            'average_agreement': avg_agreement,
            'low_agreement_examples': [i for i, a in enumerate(agreements) if a < self.min_agreement],
            'trend': self._compute_trend(),
            'recommendations': self._generate_recommendations(avg_agreement)
        }
        
        return quality_report
    
    def _compute_trend(self):
        """Compute agreement trend over time"""
        if len(self.agreement_history) < 2:
            return 'insufficient_data'
        
        recent = self.agreement_history[-5:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.01:
            return 'improving'
        elif trend < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_recommendations(self, avg_agreement):
        """Generate recommendations based on agreement score"""
        recommendations = []
        
        if avg_agreement < self.min_agreement:
            recommendations.append("Consider retraining annotators on guidelines")
            recommendations.append("Review and clarify annotation instructions")
            recommendations.append("Increase number of annotators per example")
        
        return recommendations
```

## Quality Control

### Multi-Annotator Setup

**Implementation**:
```python
class MultiAnnotatorSystem:
    def __init__(self, num_annotators_per_example=3):
        self.num_annotators = num_annotators_per_example
        self.annotators = {}
        self.annotation_queue = []
    
    def assign_annotations(self, examples):
        """
        Assign examples to annotators
        
        Args:
            examples: List of examples to annotate
        
        Returns:
            assignments: Dictionary mapping annotator_id to examples
        """
        assignments = {i: [] for i in range(len(self.annotators))}
        
        for i, example in enumerate(examples):
            # Round-robin assignment
            annotator_id = i % len(self.annotators)
            assignments[annotator_id].append(example)
        
        return assignments
    
    def aggregate_annotations(self, example_annotations):
        """
        Aggregate multiple annotations for a single example
        
        Args:
            example_annotations: List of annotations from different annotators
        
        Returns:
            aggregated_annotation: Final annotation for the example
        """
        if not example_annotations:
            return None
        
        # For binary preferences
        if all(isinstance(a, int) for a in example_annotations):
            # Majority vote
            return max(set(example_annotations), key=example_annotations.count)
        
        # For rankings
        elif all(isinstance(a, list) for a in example_annotations):
            # Borda count aggregation
            return self._borda_aggregate(example_annotations)
        
        # For ratings
        elif all(isinstance(a, (int, float)) for a in example_annotations):
            # Mean rating
            return np.mean(example_annotations)
        
        else:
            raise ValueError("Inconsistent annotation types")
    
    def _borda_aggregate(self, rankings):
        """
        Aggregate rankings using Borda count
        
        Args:
            rankings: List of rankings from different annotators
        
        Returns:
            aggregated_ranking: Final ranking
        """
        # Count votes for each position
        num_items = len(rankings[0])
        scores = {i: 0 for i in range(num_items)}
        
        for ranking in rankings:
            for position, item in enumerate(ranking):
                scores[item] += (num_items - position - 1)
        
        # Sort by scores
        aggregated_ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return aggregated_ranking
```

### Automated Quality Checks

**Implementation**:
```python
class AutomatedQualityChecker:
    def __init__(self):
        self.quality_metrics = {}
    
    def check_annotation_quality(self, annotations, metadata):
        """
        Perform automated quality checks
        
        Args:
            annotations: Annotations to check
            metadata: Additional metadata about annotations
        
        Returns:
            quality_report: Report on annotation quality
        """
        report = {
            'completion_rate': self._check_completion_rate(annotations),
            'response_time': self._check_response_time(metadata),
            'consistency': self._check_consistency(annotations),
            'bias_detection': self._check_bias(annotations, metadata),
            'recommendations': []
        }
        
        # Generate recommendations
        if report['completion_rate'] < 0.95:
            report['recommendations'].append("Low completion rate - consider simplifying interface")
        
        if report['response_time']['mean'] < 10:  # seconds
            report['recommendations'].append("Very fast responses - may indicate inattentive annotation")
        
        return report
    
    def _check_completion_rate(self, annotations):
        """Check percentage of completed annotations"""
        total = len(annotations)
        completed = sum(1 for a in annotations if a is not None)
        return completed / total if total > 0 else 0
    
    def _check_response_time(self, metadata):
        """Analyze response times"""
        times = [m.get('response_time', 0) for m in metadata]
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    def _check_consistency(self, annotations):
        """Check for annotation consistency patterns"""
        # Check for systematic biases
        if len(set(annotations)) == 1:
            return {'warning': 'All annotations identical - possible bias'}
        
        return {'status': 'normal'}
    
    def _check_bias(self, annotations, metadata):
        """Detect potential annotation bias"""
        # Check for annotator-specific patterns
        annotator_patterns = {}
        for i, (annotation, meta) in enumerate(zip(annotations, metadata)):
            annotator_id = meta.get('annotator_id')
            if annotator_id not in annotator_patterns:
                annotator_patterns[annotator_id] = []
            annotator_patterns[annotator_id].append(annotation)
        
        # Check for systematic differences between annotators
        bias_report = {}
        for annotator_id, patterns in annotator_patterns.items():
            if len(set(patterns)) == 1:
                bias_report[annotator_id] = 'Systematic bias detected'
        
        return bias_report
```

## Bias Mitigation

### Diverse Annotator Pools

**Implementation**:
```python
class DiverseAnnotatorPool:
    def __init__(self):
        self.annotators = []
        self.demographics = {}
    
    def add_annotator(self, annotator_id, demographics):
        """
        Add annotator with demographic information
        
        Args:
            annotator_id: Unique identifier for annotator
            demographics: Dictionary with demographic information
        """
        self.annotators.append(annotator_id)
        self.demographics[annotator_id] = demographics
    
    def ensure_diversity(self, required_demographics):
        """
        Ensure annotator pool meets diversity requirements
        
        Args:
            required_demographics: Dictionary of required demographic distributions
        
        Returns:
            is_diverse: Whether pool meets diversity requirements
        """
        current_demographics = self._compute_demographics()
        
        for category, required_dist in required_demographics.items():
            if category not in current_demographics:
                return False
            
            current_dist = current_demographics[category]
            for value, required_pct in required_dist.items():
                if current_dist.get(value, 0) < required_pct:
                    return False
        
        return True
    
    def _compute_demographics(self):
        """Compute current demographic distribution"""
        demographics = {}
        
        for annotator_id in self.annotators:
            annotator_demo = self.demographics[annotator_id]
            
            for category, value in annotator_demo.items():
                if category not in demographics:
                    demographics[category] = {}
                
                if value not in demographics[category]:
                    demographics[category][value] = 0
                
                demographics[category][value] += 1
        
        # Convert to percentages
        total_annotators = len(self.annotators)
        for category in demographics:
            for value in demographics[category]:
                demographics[category][value] /= total_annotators
        
        return demographics
```

### Bias Detection and Correction

**Implementation**:
```python
class BiasDetector:
    def __init__(self):
        self.bias_patterns = {}
    
    def detect_bias(self, annotations, metadata):
        """
        Detect various types of annotation bias
        
        Args:
            annotations: Annotations to analyze
            metadata: Additional metadata
        
        Returns:
            bias_report: Report on detected biases
        """
        bias_report = {
            'annotator_bias': self._detect_annotator_bias(annotations, metadata),
            'content_bias': self._detect_content_bias(annotations, metadata),
            'temporal_bias': self._detect_temporal_bias(annotations, metadata),
            'systematic_bias': self._detect_systematic_bias(annotations)
        }
        
        return bias_report
    
    def _detect_annotator_bias(self, annotations, metadata):
        """Detect bias specific to individual annotators"""
        annotator_stats = {}
        
        for annotation, meta in zip(annotations, metadata):
            annotator_id = meta.get('annotator_id')
            if annotator_id not in annotator_stats:
                annotator_stats[annotator_id] = []
            annotator_stats[annotator_id].append(annotation)
        
        bias_report = {}
        for annotator_id, stats in annotator_stats.items():
            # Check for systematic patterns
            if len(set(stats)) == 1:
                bias_report[annotator_id] = 'Systematic bias - all annotations identical'
            elif np.std(stats) < 0.1:
                bias_report[annotator_id] = 'Low variance - possible bias'
        
        return bias_report
    
    def _detect_content_bias(self, annotations, metadata):
        """Detect bias related to content characteristics"""
        content_features = [meta.get('content_features', {}) for meta in metadata]
        
        bias_report = {}
        
        # Check for bias based on content length
        lengths = [f.get('length', 0) for f in content_features]
        if lengths:
            correlation = np.corrcoef(annotations, lengths)[0, 1]
            if abs(correlation) > 0.3:
                bias_report['length_bias'] = f'Strong correlation with length: {correlation:.3f}'
        
        # Check for bias based on topic
        topics = [f.get('topic', 'unknown') for f in content_features]
        topic_annotations = {}
        for topic, annotation in zip(topics, annotations):
            if topic not in topic_annotations:
                topic_annotations[topic] = []
            topic_annotations[topic].append(annotation)
        
        for topic, topic_anns in topic_annotations.items():
            if len(topic_anns) > 10:  # Only check topics with sufficient data
                topic_mean = np.mean(topic_anns)
                overall_mean = np.mean(annotations)
                if abs(topic_mean - overall_mean) > 0.5:
                    bias_report[f'topic_bias_{topic}'] = f'Topic bias: {topic_mean:.3f} vs {overall_mean:.3f}'
        
        return bias_report
    
    def _detect_temporal_bias(self, annotations, metadata):
        """Detect bias related to timing"""
        timestamps = [meta.get('timestamp', 0) for meta in metadata]
        
        if len(set(timestamps)) > 1:
            # Check for drift over time
            correlation = np.corrcoef(annotations, timestamps)[0, 1]
            if abs(correlation) > 0.3:
                return {'temporal_drift': f'Strong correlation with time: {correlation:.3f}'}
        
        return {}
    
    def _detect_systematic_bias(self, annotations):
        """Detect systematic bias patterns"""
        # Check for ceiling/floor effects
        max_annotation = max(annotations)
        min_annotation = min(annotations)
        
        bias_report = {}
        
        if max_annotation == min_annotation:
            bias_report['no_variance'] = 'All annotations identical'
        elif np.std(annotations) < 0.1:
            bias_report['low_variance'] = 'Very low variance in annotations'
        
        # Check for distribution skew
        mean_ann = np.mean(annotations)
        median_ann = np.median(annotations)
        if abs(mean_ann - median_ann) > 0.2:
            bias_report['distribution_skew'] = f'Skewed distribution: mean={mean_ann:.3f}, median={median_ann:.3f}'
        
        return bias_report
```

## Implementation Examples

### Complete Feedback Collection Pipeline

```python
class HumanFeedbackCollector:
    def __init__(self, model, reward_model, annotator_pool):
        self.model = model
        self.reward_model = reward_model
        self.annotator_pool = annotator_pool
        self.quality_checker = AutomatedQualityChecker()
        self.bias_detector = BiasDetector()
    
    def collect_feedback_batch(self, prompts, batch_size=100):
        """
        Collect human feedback for a batch of prompts
        
        Args:
            prompts: List of input prompts
            batch_size: Number of examples to collect feedback for
        
        Returns:
            feedback_data: Collected feedback data
        """
        # Generate responses
        responses = self._generate_responses(prompts)
        
        # Select examples for annotation
        selected_examples = self._select_examples(prompts, responses)
        
        # Assign to annotators
        assignments = self.annotator_pool.assign_annotations(selected_examples)
        
        # Collect annotations
        raw_annotations = self._collect_annotations(assignments)
        
        # Quality control
        quality_report = self.quality_checker.check_annotation_quality(raw_annotations)
        bias_report = self.bias_detector.detect_bias(raw_annotations)
        
        # Aggregate annotations
        aggregated_annotations = self._aggregate_annotations(raw_annotations)
        
        # Filter high-quality examples
        final_data = self._filter_high_quality(aggregated_annotations, quality_report)
        
        return {
            'feedback_data': final_data,
            'quality_report': quality_report,
            'bias_report': bias_report
        }
    
    def _generate_responses(self, prompts):
        """Generate multiple responses for each prompt"""
        responses = []
        
        for prompt in prompts:
            # Generate multiple responses with different sampling strategies
            response_set = []
            
            # Greedy response
            greedy_response = self.model.generate(prompt, max_length=100, do_sample=False)
            response_set.append(greedy_response)
            
            # Sampled responses
            for _ in range(2):
                sampled_response = self.model.generate(prompt, max_length=100, do_sample=True, temperature=0.7)
                response_set.append(sampled_response)
            
            responses.append(response_set)
        
        return responses
    
    def _select_examples(self, prompts, responses):
        """Select examples for annotation using active learning"""
        # Compute uncertainty scores
        uncertainties = []
        for prompt, response_set in zip(prompts, responses):
            # Use reward model to estimate uncertainty
            rewards = [self.reward_model.predict(prompt, resp) for resp in response_set]
            uncertainty = np.std(rewards)  # Higher std = more uncertain
            uncertainties.append(uncertainty)
        
        # Select examples with highest uncertainty
        sorted_indices = np.argsort(uncertainties)[::-1]
        selected_indices = sorted_indices[:len(prompts)//2]  # Select top 50%
        
        return [(prompts[i], responses[i]) for i in selected_indices]
    
    def _collect_annotations(self, assignments):
        """Collect annotations from human annotators"""
        # This would interface with actual annotation platform
        # For demonstration, we'll simulate annotations
        
        annotations = []
        for annotator_id, examples in assignments.items():
            for example in examples:
                # Simulate annotation process
                annotation = self._simulate_annotation(example)
                annotations.append({
                    'annotator_id': annotator_id,
                    'example': example,
                    'annotation': annotation,
                    'timestamp': time.time()
                })
        
        return annotations
    
    def _simulate_annotation(self, example):
        """Simulate human annotation (replace with actual annotation interface)"""
        prompt, responses = example
        
        # Simulate preference annotation
        # In practice, this would be done by human annotators
        if len(responses) == 2:
            # Binary preference
            return np.random.choice([0, 1])
        else:
            # Ranking
            return list(np.random.permutation(len(responses)))
    
    def _aggregate_annotations(self, raw_annotations):
        """Aggregate multiple annotations for each example"""
        # Group by example
        example_annotations = {}
        for ann in raw_annotations:
            example_key = str(ann['example'])
            if example_key not in example_annotations:
                example_annotations[example_key] = []
            example_annotations[example_key].append(ann['annotation'])
        
        # Aggregate each example
        aggregated = {}
        for example_key, annotations in example_annotations.items():
            aggregated[example_key] = self.annotator_pool.aggregate_annotations(annotations)
        
        return aggregated
    
    def _filter_high_quality(self, aggregated_annotations, quality_report):
        """Filter out low-quality annotations"""
        high_quality_data = []
        
        for example_key, annotation in aggregated_annotations.items():
            # Check if this example has sufficient agreement
            if example_key in quality_report.get('high_agreement_examples', []):
                high_quality_data.append({
                    'example': eval(example_key),  # Convert back to tuple
                    'annotation': annotation
                })
        
        return high_quality_data
```

## Advanced Techniques

### Active Learning for Feedback Collection

```python
class ActiveLearningFeedbackCollector:
    def __init__(self, model, reward_model, uncertainty_threshold=0.3):
        self.model = model
        self.reward_model = reward_model
        self.uncertainty_threshold = uncertainty_threshold
        self.collected_feedback = []
    
    def select_informative_examples(self, candidate_prompts, batch_size=50):
        """
        Select most informative examples for annotation
        
        Args:
            candidate_prompts: Pool of candidate prompts
            batch_size: Number of examples to select
        
        Returns:
            selected_examples: Most informative examples
        """
        # Generate responses for all candidates
        all_responses = self._generate_diverse_responses(candidate_prompts)
        
        # Compute uncertainty scores
        uncertainties = []
        for prompt, responses in zip(candidate_prompts, all_responses):
            uncertainty = self._compute_uncertainty(prompt, responses)
            uncertainties.append(uncertainty)
        
        # Select examples with highest uncertainty
        sorted_indices = np.argsort(uncertainties)[::-1]
        selected_indices = sorted_indices[:batch_size]
        
        selected_examples = []
        for idx in selected_indices:
            selected_examples.append({
                'prompt': candidate_prompts[idx],
                'responses': all_responses[idx],
                'uncertainty': uncertainties[idx]
            })
        
        return selected_examples
    
    def _compute_uncertainty(self, prompt, responses):
        """
        Compute uncertainty for a prompt-response set
        
        Args:
            prompt: Input prompt
            responses: List of generated responses
        
        Returns:
            uncertainty: Uncertainty score
        """
        # Get reward model predictions
        rewards = [self.reward_model.predict(prompt, resp) for resp in responses]
        
        # Compute uncertainty as standard deviation of rewards
        uncertainty = np.std(rewards)
        
        # Add diversity penalty
        diversity_penalty = self._compute_diversity_penalty(responses)
        
        return uncertainty + diversity_penalty
    
    def _compute_diversity_penalty(self, responses):
        """Compute diversity penalty to encourage diverse responses"""
        if len(responses) < 2:
            return 0
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                similarity = self._compute_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        # Diversity penalty: lower average similarity = higher diversity
        avg_similarity = np.mean(similarities)
        diversity_penalty = -avg_similarity  # Negative because we want high diversity
        
        return diversity_penalty
    
    def _compute_similarity(self, resp1, resp2):
        """Compute similarity between two responses"""
        # Simple token overlap similarity
        tokens1 = set(resp1.split())
        tokens2 = set(resp2.split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0
```

### Multi-Modal Feedback Collection

```python
class MultiModalFeedbackCollector:
    def __init__(self):
        self.feedback_types = ['preference', 'rating', 'explanation']
    
    def collect_multimodal_feedback(self, examples):
        """
        Collect multiple types of feedback for each example
        
        Args:
            examples: List of examples to collect feedback for
        
        Returns:
            multimodal_feedback: Dictionary with different feedback types
        """
        feedback_data = {
            'preferences': [],
            'ratings': [],
            'explanations': []
        }
        
        for example in examples:
            # Collect preference feedback
            preference = self._collect_preference_feedback(example)
            feedback_data['preferences'].append(preference)
            
            # Collect rating feedback
            rating = self._collect_rating_feedback(example)
            feedback_data['ratings'].append(rating)
            
            # Collect explanation feedback
            explanation = self._collect_explanation_feedback(example)
            feedback_data['explanations'].append(explanation)
        
        return feedback_data
    
    def _collect_preference_feedback(self, example):
        """Collect binary preference feedback"""
        prompt, responses = example
        
        # Present responses to annotator
        print(f"Prompt: {prompt}")
        print(f"Response A: {responses[0]}")
        print(f"Response B: {responses[1]}")
        print("Which response do you prefer? (A/B)")
        
        # In practice, this would be a web interface
        preference = input().upper()
        
        return 0 if preference == 'A' else 1
    
    def _collect_rating_feedback(self, example):
        """Collect rating feedback"""
        prompt, responses = example
        
        ratings = []
        for i, response in enumerate(responses):
            print(f"Prompt: {prompt}")
            print(f"Response {i+1}: {response}")
            print("Rate this response (1-5):")
            
            rating = int(input())
            ratings.append(rating)
        
        return ratings
    
    def _collect_explanation_feedback(self, example):
        """Collect natural language explanation"""
        prompt, responses = example
        
        print(f"Prompt: {prompt}")
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response}")
        
        print("Please explain your reasoning for your preference:")
        explanation = input()
        
        return explanation
```

## Best Practices

### 1. Clear Annotation Instructions

- **Specific Criteria**: Define exactly what to evaluate
- **Examples**: Provide clear examples of good and bad annotations
- **Edge Cases**: Address common edge cases and ambiguities
- **Iterative Refinement**: Update instructions based on feedback

### 2. Quality Control

- **Multi-Annotator Setup**: Use multiple annotators per example
- **Agreement Monitoring**: Track inter-annotator agreement
- **Consistency Checks**: Identify and address systematic biases
- **Automated Screening**: Use automated tools to flag potential issues

### 3. Bias Mitigation

- **Diverse Annotator Pool**: Ensure representation across demographics
- **Bias Detection**: Monitor for systematic biases
- **Bias Correction**: Implement strategies to address detected biases
- **Transparency**: Document potential biases and limitations

### 4. Efficient Collection

- **Active Learning**: Focus on most informative examples
- **Batch Processing**: Collect feedback in batches for efficiency
- **Quality vs. Quantity**: Prioritize high-quality feedback over large volumes
- **Iterative Improvement**: Continuously refine collection strategies

### 5. Data Management

- **Version Control**: Track changes to annotation guidelines
- **Metadata Tracking**: Record annotator demographics and context
- **Quality Metrics**: Monitor and report on data quality
- **Documentation**: Maintain clear documentation of collection process

## Summary

Human feedback collection is a critical component of RLHF systems that requires careful attention to:

1. **Diverse Feedback Types**: Binary preferences, rankings, ratings, and natural language explanations
2. **Quality Control**: Multi-annotator setups, agreement monitoring, and automated screening
3. **Bias Mitigation**: Diverse annotator pools and systematic bias detection
4. **Efficient Collection**: Active learning and strategic example selection
5. **Best Practices**: Clear guidelines, iterative refinement, and comprehensive documentation

Effective feedback collection enables the training of language models that better align with human values and preferences, ultimately leading to more useful, safe, and honest AI systems.

---

**Note**: This guide provides the theoretical and practical foundations for human feedback collection. For specific implementation details and platform integrations, refer to the implementation examples and external resources referenced in the main README.

## From Data Collection to Reward Modeling

We've now explored **human feedback collection** - the systematic process of gathering, structuring, and validating human preferences and judgments about model outputs. We've seen how different types of feedback (binary preferences, rankings, ratings, natural language explanations) capture different aspects of human judgment, how to design effective annotation guidelines and quality control measures, and how to mitigate biases and ensure diverse perspectives in the feedback collection process.

However, while collecting high-quality human feedback is essential, **raw feedback data** is not directly usable for training language models. Consider having thousands of preference judgments - we need a way to convert these relative preferences into a reward function that can guide policy optimization and provide consistent feedback during training.

This motivates our exploration of **reward modeling** - the process of learning a function that maps prompt-response pairs to scalar reward values, capturing human preferences and judgments. We'll see how to design neural network architectures that can learn from preference data, how to formulate training objectives that capture the relative nature of human preferences, how to validate and calibrate reward models, and how to address challenges like reward hacking and distributional shift.

The transition from human feedback collection to reward modeling represents the bridge from raw preference data to learnable reward signals - taking our understanding of how to collect human feedback and applying it to the challenge of building reward functions that can guide effective policy optimization.

In the next section, we'll explore reward modeling, understanding how to convert human preferences into reward functions that enable successful RLHF training.

---

**Previous: [Fundamentals of RL for Language Models](01_fundamentals_of_rl_for_language_models.md)** - Understand the mathematical foundations of RLHF.

**Next: [Reward Modeling](03_reward_modeling.md)** - Learn how to convert human preferences into reward functions. 