import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from urllib.request import urlopen
from io import StringIO

# Download UCI Spambase dataset
# This dataset contains features extracted from emails and a label (1=spam, 0=not spam)
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
col_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'

# Load the dataset into a pandas DataFrame
# There are 57 features and 1 label column (no header in the CSV)
data = pd.read_csv(data_url, header=None)
# The last column is the label: 1 for spam, 0 for not spam
X = data.iloc[:, :-1]  # Features (word frequencies, etc.)
y = data.iloc[:, -1]   # Labels (spam or not spam)

# Define the different training set sizes to evaluate learning curves
train_sizes = [50, 100, 200, 400, 750, 1000, 1250, 1500, 1750, 2000]
nb_errors = []  # To store error rates for Naive Bayes
svm_errors = [] # To store error rates for SVM

# Split the data into a fixed training set and test set (80% train, 20% test)
# Stratify ensures the same proportion of spam/not-spam in both sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# For each training set size, train and evaluate both classifiers
for size in train_sizes:
    # Select the first 'size' examples from the training set
    X_train = X_train_full.iloc[:size, :]
    y_train = y_train_full.iloc[:size]

    # --- Naive Bayes Classifier ---
    # MultinomialNB is suitable for word count features (like spam data)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)  # Train on the subset
    y_pred_nb = nb.predict(X_test)  # Predict on the fixed test set
    nb_error = 1 - accuracy_score(y_test, y_pred_nb)  # Compute error rate
    nb_errors.append(nb_error)

    # --- Support Vector Machine Classifier ---
    # Linear kernel is used for speed and because text data is often linearly separable
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)  # Train on the subset
    y_pred_svm = svm.predict(X_test)  # Predict on the fixed test set
    svm_error = 1 - accuracy_score(y_test, y_pred_svm)  # Compute error rate
    svm_errors.append(svm_error)

# --- Plotting the Results ---
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, nb_errors, marker='o', label='Naive Bayes')
plt.plot(train_sizes, svm_errors, marker='s', label='SVM (linear)')
plt.xlabel('Number of Training Examples')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Number of Training Examples (UCI Spambase)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print error rates for reference ---
print('Training Size\tNaive Bayes Error\tSVM Error')
for size, nb_e, svm_e in zip(train_sizes, nb_errors, svm_errors):
    print(f'{size}\t\t{nb_e:.3f}\t\t\t{svm_e:.3f}') 