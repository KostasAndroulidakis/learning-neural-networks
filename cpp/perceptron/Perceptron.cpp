#include "Perceptron.h"
#include <algorithm>
#include <numeric>
#include <cmath>

Perceptron::Perceptron(int n_inputs, double learning_rate, int max_epochs, int random_state)
    : n_inputs_(n_inputs), 
      learning_rate_(learning_rate), 
      max_epochs_(max_epochs),
      random_state_(random_state),
      bias_(0.0) 
{
    // Initialize random number generator with seed
    rng_.seed(random_state_);
    
    // Initialize weights with small random values
    std::normal_distribution<double> dist(0.0, 0.01);
    weights_.resize(n_inputs_);
    for (auto& w : weights_) {
        w = dist(rng_);
    }
}

Perceptron& Perceptron::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // Reset errors list
    errors_.clear();
    
    // Training loop
    for (int epoch = 0; epoch < max_epochs_; ++epoch) {
        // Make predictions
        std::vector<int> y_pred = predict(X);
        
        // Update weights and get misclassification count
        int misclassified = update_weights(X, y, y_pred);
        errors_.push_back(misclassified);
        
        // Stop training if all samples are correctly classified
        if (misclassified == 0) {
            break;
        }
    }
    
    return *this;
}

std::vector<int> Perceptron::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& x : X) {
        // Calculate net input: w·x + b
        double net_input = bias_;
        for (size_t i = 0; i < x.size(); ++i) {
            net_input += weights_[i] * x[i];
        }
        
        // Apply step activation function
        predictions.push_back(net_input >= 0 ? 1 : 0);
    }
    
    return predictions;
}

int Perceptron::update_weights(const std::vector<std::vector<double>>& X, 
                               const std::vector<int>& y, 
                               const std::vector<int>& y_pred) {
    int misclassified = 0;
    
    // Update for each sample
    for (size_t i = 0; i < X.size(); ++i) {
        // Calculate error
        int error = y[i] - y_pred[i];
        
        // Update only if there's an error
        if (error != 0) {
            misclassified++;
            
            // Update bias
            bias_ += learning_rate_ * error;
            
            // Update weights
            for (size_t j = 0; j < weights_.size(); ++j) {
                weights_[j] += learning_rate_ * error * X[i][j];
            }
        }
    }
    
    return misclassified;
}

const std::vector<int>& Perceptron::get_errors() const {
    return errors_;
}

const std::vector<double>& Perceptron::get_weights() const {
    return weights_;
}

double Perceptron::get_bias() const {
    return bias_;
}
