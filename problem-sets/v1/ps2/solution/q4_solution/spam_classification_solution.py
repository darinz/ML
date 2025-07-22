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
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
col_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'

# Load data
data = pd.read_csv(data_url, header=None)
# The last column is the label: 1 for spam, 0 for not spam
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Define training set sizes to evaluate
train_sizes = [50, 100, 200, 400, 750, 1000, 1250, 1500, 1750, 2000]
nb_errors = []
svm_errors = []

# Use a fixed test set (20% of the data)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for size in train_sizes:
    # Sample a subset of the training data
    X_train = X_train_full.iloc[:size, :]
    y_train = y_train_full.iloc[:size]

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    nb_error = 1 - accuracy_score(y_test, y_pred_nb)
    nb_errors.append(nb_error)

    # SVM (linear kernel for speed)
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_error = 1 - accuracy_score(y_test, y_pred_svm)
    svm_errors.append(svm_error)

# Plot error rates
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

# Print error rates for reference
print('Training Size\tNaive Bayes Error\tSVM Error')
for size, nb_e, svm_e in zip(train_sizes, nb_errors, svm_errors):
    print(f'{size}\t\t{nb_e:.3f}\t\t\t{svm_e:.3f}') 