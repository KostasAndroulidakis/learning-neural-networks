#ifndef DATA_HANDLER_H
#define DATA_HANDLER_H

#include <vector>
#include <random>
#include <algorithm>
#include <utility>

// Structure to hold dataset features and targets
struct Dataset {
    std::vector<std::vector<double>> X;  // Features
    std::vector<int> y;                  // Targets
};

// Structure to hold train/test split
struct TrainTestSplit {
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> X_test;
    std::vector<int> y_train;
    std::vector<int> y_test;
};

// Generate a synthetic linearly separable dataset
Dataset generate_linearly_separable_data(int n_samples = 100, int n_features = 2, 
                                          double noise = 0.1, int random_state = 0) {
    Dataset dataset;
    std::mt19937 rng(random_state);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // Generate random weight vector to define decision boundary
    std::vector<double> weights(n_features);
    for (auto& w : weights) {
        w = normal_dist(rng);
    }
    
    // Generate random data points
    dataset.X.resize(n_samples, std::vector<double>(n_features));
    for (auto& sample : dataset.X) {
        for (auto& feature : sample) {
            feature = normal_dist(rng);
        }
    }
    
    // Compute raw output: w·x (no bias for simplicity)
    dataset.y.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double raw_output = 0.0;
        for (int j = 0; j < n_features; ++j) {
            raw_output += dataset.X[i][j] * weights[j];
        }
        
        // Add noise
        if (noise > 0) {
            raw_output += normal_dist(rng) * noise;
        }
        
        // Create binary target: 1 if w·x > 0, 0 otherwise
        dataset.y[i] = raw_output > 0 ? 1 : 0;
    }
    
    return dataset;
}

// Split dataset into training and test sets
TrainTestSplit train_test_split(const std::vector<std::vector<double>>& X, 
                                const std::vector<int>& y, 
                                double test_size = 0.2, 
                                int random_state = 0) {
    TrainTestSplit split;
    std::mt19937 rng(random_state);
    
    int n_samples = static_cast<int>(X.size());
    int test_samples = static_cast<int>(n_samples * test_size);
    
    // Create a vector of indices and shuffle it
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Split indices into test and train sets
    std::vector<int> test_indices(indices.begin(), indices.begin() + test_samples);
    std::vector<int> train_indices(indices.begin() + test_samples, indices.end());
    
    // Allocate space for the split data
    split.X_train.resize(train_indices.size());
    split.X_test.resize(test_indices.size());
    split.y_train.resize(train_indices.size());
    split.y_test.resize(test_indices.size());
    
    // Fill the train set
    for (size_t i = 0; i < train_indices.size(); ++i) {
        int idx = train_indices[i];
        split.X_train[i] = X[idx];
        split.y_train[i] = y[idx];
    }
    
    // Fill the test set
    for (size_t i = 0; i < test_indices.size(); ++i) {
        int idx = test_indices[i];
        split.X_test[i] = X[idx];
        split.y_test[i] = y[idx];
    }
    
    return split;
}

// Normalize features using min-max scaling
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
normalize_data(const std::vector<std::vector<double>>& X_train, 
               const std::vector<std::vector<double>>& X_test = std::vector<std::vector<double>>()) {
    if (X_train.empty()) {
        return {X_train, X_test};
    }
    
    int n_features = X_train[0].size();
    
    // Find min and max for each feature
    std::vector<double> X_min(n_features, std::numeric_limits<double>::max());
    std::vector<double> X_max(n_features, std::numeric_limits<double>::lowest());
    
    for (const auto& sample : X_train) {
        for (int j = 0; j < n_features; ++j) {
            X_min[j] = std::min(X_min[j], sample[j]);
            X_max[j] = std::max(X_max[j], sample[j]);
        }
    }
    
    // Calculate range and handle division by zero
    std::vector<double> X_range(n_features);
    for (int j = 0; j < n_features; ++j) {
        X_range[j] = X_max[j] - X_min[j];
        if (X_range[j] == 0) {
            X_range[j] = 1.0;  // Avoid division by zero
        }
    }
    
    // Normalize training data
    std::vector<std::vector<double>> X_train_normalized = X_train;
    for (auto& sample : X_train_normalized) {
        for (int j = 0; j < n_features; ++j) {
            sample[j] = (sample[j] - X_min[j]) / X_range[j];
        }
    }
    
    // Normalize test data if provided
    std::vector<std::vector<double>> X_test_normalized;
    if (!X_test.empty()) {
        X_test_normalized = X_test;
        for (auto& sample : X_test_normalized) {
            for (int j = 0; j < n_features; ++j) {
                sample[j] = (sample[j] - X_min[j]) / X_range[j];
            }
        }
    }
    
    return {X_train_normalized, X_test_normalized};
}

#endif // DATA_HANDLER_H
