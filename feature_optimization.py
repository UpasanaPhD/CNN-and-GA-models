

# Commented out IPython magic to ensure Python compatibility.
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import Model
from keras import applications
from sklearn import metrics
from random import randint
import tensorflow as tf
from sklearn import svm
# %matplotlib inline
import warnings
import seaborn as sns
import itertools
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset and labels here (replace with your data)
data = pd.read_csv("csv-files-for-ga/500_new2.csv")
labels = data["Finding Labels"]
data.drop(["Finding Labels"], axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Genetic Algorithm Parameters
n_population = 80  # Number of chromosomes in the population
n_generations = 500  # Number of generations
n_genes = X_train.shape[1]  # Number of features (genes)
n_parents = 50  # Number of parents to select for breeding
mutation_rate = 0.02  # Probability of a gene mutating

# Fixed Crossover Rate
crossover_rate = 0.98
genes_to_inherit = int(crossover_rate * n_genes)

# Initialize the population
population = np.random.choice([0, 1], size=(n_population, n_genes))

# Define the fitness function (accuracy of RandomForestClassifier)
def fitness(chromosome):
    selected_features = X_train.columns[chromosome == 1]
    if len(selected_features) == 0:
        return 0  # Avoid fitness of 0 for empty chromosome
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train[selected_features], y_train)
    y_pred = clf.predict(X_test[selected_features])
    return accuracy_score(y_test, y_pred)

# Main Genetic Algorithm Loop
for generation in range(n_generations):
    # Calculate fitness scores for all chromosomes in the population
    fitness_scores = np.array([fitness(chromosome) for chromosome in population])

    # Select the top-performing parents based on fitness scores
    parents = population[np.argsort(fitness_scores)[-n_parents:]]

    # Create an empty population for the next generation
    new_population = np.empty_like(population)

    # Shuffle the parents for pairing up for crossover
    np.random.shuffle(parents)

    # Crossover: Create offspring from pairs of parents
    for i in range(0, n_parents, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        new_population[i, :genes_to_inherit] = parent1[:genes_to_inherit]
        new_population[i, genes_to_inherit:] = parent2[genes_to_inherit:]
        new_population[i + 1, :genes_to_inherit] = parent2[:genes_to_inherit]
        new_population[i + 1, genes_to_inherit:] = parent1[genes_to_inherit:]

    # Mutation: Apply mutations to the new population
    for i in range(n_population):
        mutation_mask = np.random.rand(n_genes) < mutation_rate
        new_population[i, mutation_mask] = 1 - new_population[i, mutation_mask]

    # Replace the old population with the new population
    population = new_population

# Find the best chromosome (feature selection)
best_chromosome = population[np.argmax(fitness_scores)]

# Print the selected features and their indices
selected_features = X_train.columns[best_chromosome == 1]
print("Selected Features:", selected_features)
print("Number of Selected Features:", len(selected_features))

# Train a final model with the selected features and evaluate on the test set
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train[selected_features], y_train)
y_pred = clf.predict(X_test[selected_features])
accuracy = accuracy_score(y_test, y_pred)
print("Final Model Accuracy:", accuracy)

# Print the classification report
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))

import time
from sklearn.metrics import classification_report, accuracy_score

# Print classification report and prediction time
final_chromosome = chromo_df_bc[-1]
logmodel2.fit(X_train.iloc[:, final_chromosome], Y_train)

start_time = time.time()  # Record start time
predictions = logmodel2.predict(X_test.iloc[:, final_chromosome])
end_time = time.time()  # Record end time
prediction_time = end_time - start_time

accuracy = accuracy_score(Y_test, predictions)

print("Classification Report:")
print(classification_report(Y_test, predictions))
print("Accuracy:", accuracy)
print("Prediction Time:", prediction_time, "seconds")

