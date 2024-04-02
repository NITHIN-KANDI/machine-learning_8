import numpy as np
from collections import Counter

def calculate_entropy(labels):
    # Calculate the entropy of a set of labels
    class_counts = np.bincount(labels)
    probabilities = class_counts / len(labels)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(features, labels, feature_index):
    # Calculate the information gain of a feature
    total_entropy = calculate_entropy(labels)
    values, counts = np.unique(features[:, feature_index], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / len(features)) * calculate_entropy(labels[features[:, feature_index] == values[i]]) for i in range(len(values))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_node(X, y):
    # Find the feature index with the highest information gain
    max_information_gain = -1
    root_feature_index = -1
    for i in range(X.shape[1]):
        information_gain = calculate_information_gain(X, y, i)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            root_feature_index = i
    return root_feature_index

def equal_width_binning(data, num_bins):
    # Perform equal width binning on a continuous feature
    min_value = np.min(data.astype(float))
    max_value = np.max(data.astype(float))
    bin_width = (max_value - min_value) / num_bins
    bins = [min_value + i * bin_width for i in range(num_bins + 1)]
    binned_data = np.digitize(data, bins) - 1
    return binned_data

def frequency_binning(data, num_bins):
    # Perform frequency binning on a continuous feature
    counts, bins = np.histogram(data.astype(float), bins=num_bins)
    binned_data = np.digitize(data, bins[:-1])
    return binned_data

def bin_continuous_feature(X, feature_index, binning_type='equal_width', num_bins=5):
    # Bin a continuous feature based on the specified binning type
    data = X[:, feature_index]
    if np.issubdtype(data.dtype, np.number):
        if binning_type == 'equal_width':
            return equal_width_binning(data, num_bins)
        elif binning_type == 'frequency':
            return frequency_binning(data, num_bins)
        else:
            raise ValueError("Invalid binning type. Please choose 'equal_width' or 'frequency'.")
    else:
        return data 

# Example usage:
X = np.array([
    [1, 3.5, 'X'],
    [2, 2.5, 'Y'],
    [3, 4.5, 'Y'],
    [4, 3.0, 'X'],
    [5, 4.0, 'X']
])
y = np.array([0, 1, 1, 0, 0])

# Binning continuous feature 'A1' using equal width binning with 3 bins
binned_data = bin_continuous_feature(X, 1, binning_type='equal_width', num_bins=3)
print("Binned feature values (Equal Width Binning):", binned_data)

# Binning continuous feature 'A1' using frequency binning with 3 bins
binned_data = bin_continuous_feature(X, 1, binning_type='frequency', num_bins=3)
print("Binned feature values (Frequency Binning):", binned_data)
