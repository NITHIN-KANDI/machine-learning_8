import numpy as np
from collections import Counter

def calculate_entropy(labels):
    # Count occurrences of each class
    class_counts = np.bincount(labels)
    # Calculate probabilities
    probabilities = class_counts / len(labels)
    # Calculate entropy
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(features, labels, feature_index):
    # Calculate total entropy of the dataset
    total_entropy = calculate_entropy(labels)
    
    # Extract unique values and their counts for the specified feature
    unique_values, value_counts = np.unique(features[:, feature_index], return_counts=True)
    
    # Calculate the weighted entropy for each value of the feature
    weighted_entropy = np.sum([(value_counts[i] / len(features)) * calculate_entropy(labels[features[:, feature_index] == unique_values[i]]) for i in range(len(unique_values))])
    
    # Calculate information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_feature_value(features, labels):
    max_information_gain = -1
    root_feature_value = None
    
    # Iterate over each feature to find the one with maximum information gain
    for i in range(features.shape[1]):
        information_gain = calculate_information_gain(features, labels, i)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            root_feature_value = features[0, i]  # Set root feature value to the value of the first instance
    
    return root_feature_value


# Example usage:
# Define your dataset
features = np.array([
    [8, 'A', 'X'],
    [7, 'C', 'Y'],
    [9, 'A', 'Y'],
    [4, 'B', 'X'],
    [5, 'B', 'X']
])
labels = np.array([2, 7, 0, 6, 3])

# Find the root node feature value
root_feature_value = find_root_feature_value(features, labels)
print("Root feature value:", root_feature_value)
