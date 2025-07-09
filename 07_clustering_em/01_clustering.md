# Clustering and the $`k`$-means Algorithm

Clustering is a fundamental task in unsupervised machine learning. Imagine you have a collection of objects (like photos, documents, or customer profiles) and you want to group them so that similar objects are in the same group, but you don't know in advance what those groups should be. This is the essence of clustering: discovering structure in data without any labels.

## What is Clustering? (Motivation & Analogy)

Suppose you walk into a room full of people and want to form groups based on how similar they look or behave, but you have no idea what the groups should be. You might start by guessing some group centers, then assign people to the nearest group, and then adjust the group centers based on who joined. This is exactly what the $`k`$-means algorithm does, but with data points instead of people.

- **Supervised vs. Unsupervised Learning:**
  - *Supervised learning* uses labeled data (e.g., images with tags like "cat" or "dog").
  - *Unsupervised learning* (like clustering) uses only the features of the data, with no labels, to find patterns or groupings.

## The $`k`$-means Clustering Algorithm (Step-by-Step)

The $`k`$-means algorithm partitions a dataset into $`k`$ clusters, where each cluster is represented by its centroid (center point). The goal is to minimize the total distance between each point and its assigned centroid.

### Step 1: Initialize Centroids

Pick $`k`$ initial centroids $`\mu_1, \mu_2, \ldots, \mu_k \in \mathbb{R}^d`$ randomly from the data points.
- **Notation:**
  - $`x^{(i)}`$ is the $`i`$-th data point (a vector in $`\mathbb{R}^d`$).
  - $`\mu_j`$ is the centroid of cluster $`j`$.
- **Why random?**
  - We don't know where the clusters are, so we start with a guess. Picking actual data points helps ensure centroids are in the right space.

### Step 2: Assignment Step (Cluster Assignment)

For each data point $`x^{(i)}`$, assign it to the cluster with the closest centroid:

```math
c^{(i)} := \arg\min_j \|x^{(i)} - \mu_j\|^2.
```
- **What does this mean?**
  - For each $`x^{(i)}`$, compute the squared Euclidean distance to each centroid $`\mu_j`$.
  - Assign $`x^{(i)}`$ to the cluster $`j`$ with the smallest distance.
- **Geometric intuition:**
  - Imagine drawing circles around each centroid; each point joins the group whose circle it falls into first.

### Step 3: Update Step (Move Centroids)

For each cluster $`j`$, update its centroid to be the mean of all points assigned to it:

```math
\mu_j := \frac{\sum_{i=1}^n 1\{c^{(i)} = j\} x^{(i)}}{\sum_{i=1}^n 1\{c^{(i)} = j\}}.
```
- **What does this mean?**
  - $`1\{c^{(i)} = j\}`$ is 1 if $`x^{(i)}`$ is assigned to cluster $`j`$, 0 otherwise.
  - The numerator sums all points in cluster $`j`$; the denominator counts how many points are in cluster $`j`$.
  - The new centroid is the average (mean) of its assigned points.
- **Why the mean?**
  - The mean minimizes the sum of squared distances within the cluster (see below for math intuition).

### Step 4: Repeat Until Convergence

Repeat the assignment and update steps until the assignments stop changing (or the centroids move very little).
- **Convergence:**
  - The algorithm is guaranteed to stop because there are only finitely many ways to assign points to clusters, and each step never increases the total cost (see below).

### Worked Example (with Small Numbers)

Suppose you have 6 points on a line: 1, 2, 3, 10, 11, 12, and you want $`k=2`$ clusters.
- Initialize centroids: $`\mu_1 = 2`$, $`\mu_2 = 11`$.
- Assignment: Points 1, 2, 3 are closer to 2; points 10, 11, 12 are closer to 11.
- Update: $`\mu_1 = (1+2+3)/3 = 2`$, $`\mu_2 = (10+11+12)/3 = 11`$.
- No change, so the algorithm stops.

## The Distortion (Cost) Function: Why Does k-means Work?

The $`k`$-means algorithm minimizes the **distortion function** (also called the cost or objective function):

```math
J(c, \mu) = \sum_{i=1}^n \|x^{(i)} - \mu_{c^{(i)}}\|^2
```
- $`J`$ is the total squared distance from each point to its assigned centroid.
- The algorithm alternates between minimizing $`J`$ with respect to the assignments $`c`$ (assignment step) and the centroids $`\mu`$ (update step).
- **Why does this work?**
  - Each step either decreases $`J`$ or leaves it unchanged.
  - Since there are only finitely many possible assignments, the process must eventually stop.

### Why the Mean Minimizes Squared Distance

Suppose you have points $`x_1, x_2, ..., x_m`$ and want to pick a single point $`\mu`$ to minimize $`\sum_{i=1}^m \|x_i - \mu\|^2`$. The solution is $`\mu = \frac{1}{m} \sum_{i=1}^m x_i`$ (the mean). This is why the update step uses the mean.

## Local Minima and Multiple Runs

- The distortion function $`J`$ is **non-convex**: it can have many local minima (bad solutions).
- $`k`$-means can get stuck in a local minimum depending on the initial centroids.
- **Practical tip:** Run $`k`$-means several times with different random initializations and pick the result with the lowest $`J`$.
- **Visualization:** Imagine a landscape with many valleys; the algorithm always goes downhill, but may not reach the lowest valley.

## When to Use (and Not Use) k-means

**Use k-means when:**
- You want to partition data into spherical, equally-sized clusters.
- The clusters are well-separated in feature space.
- You want a fast, simple algorithm.

**Avoid k-means when:**
- Clusters are not spherical (e.g., elongated or of different sizes).
- There are many outliers or noise.
- The number of clusters $`k`$ is not known or is hard to choose.

## Frequently Asked Questions (FAQ)

**Q: How do I choose $`k`$?**
- Try different values and use the "elbow method": plot $`J`$ vs. $`k`$ and look for a point where the decrease slows down.

**Q: What distance metric does k-means use?**
- Standard k-means uses Euclidean distance. Other variants use different metrics.

**Q: What if a cluster gets no points?**
- Reinitialize its centroid randomly.

**Q: Is k-means sensitive to outliers?**
- Yes, because the mean is affected by extreme values.

## Summary

- $`k`$-means is a simple, widely used clustering algorithm.
- It alternates between assigning points to the nearest centroid and updating centroids to the mean of their assigned points.
- It minimizes the sum of squared distances (distortion) and is guaranteed to converge to a local minimum.
- Multiple runs and careful initialization can improve results.
- Understanding the math and intuition behind each step helps you use k-means effectively.

