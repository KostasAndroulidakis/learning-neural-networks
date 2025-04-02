#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <random>
#include <functional>

class Perceptron {
public:
    // Constructor
    Perceptron(int n_inputs, double learning_rate = 0.01, int max_epochs = 1000, int random_state = 0);
    
    // Train the perceptron with given inputs and targets
    Perceptron& fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    
    // Make predictions for given inputs
    std::vector<int> predict(const std::vector<std::vector<double>>& X);
    
    // Return the number of misclassifications for each epoch
    const std::vector<int>& get_errors() const;
    
    // Access model parameters
    const std::vector<double>& get_weights() const;
    double get_bias() const;
    
private:
    int n_inputs_;                     // Number of input features
    double learning_rate_;             // Learning rate (eta)
    int max_epochs_;                   // Maximum number of training epochs
    int random_state_;                 // Seed for random number generator
    
    std::vector<double> weights_;      // Weight vector
    double bias_;                      // Bias term
    std::vector<int> errors_;          // Number of misclassifications in each epoch
    
    // Random number generation
    std::mt19937 rng_;
    
    // Update weights based on prediction errors
    int update_weights(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<int>& y_pred);
};

#endif // PERCEPTRON_H
