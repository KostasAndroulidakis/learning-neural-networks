#include <iostream>
#include <vector>
#include "Perceptron.h"
#include "activation.h"

int main() {
    std::cout << "=== AND Gate Implementation with Perceptron ===" << std::endl << std::endl;
    
    // Create training data for AND gate
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    std::vector<int> y = {0, 0, 0, 1};  // AND gate truth table
    
    // Initialize perceptron with 2 inputs
    Perceptron perceptron(2, 0.1, 100, 42);
    
    // Train the perceptron
    std::cout << "Training perceptron on AND gate..." << std::endl;
    perceptron.fit(X, y);
    
    // Display learned weights and bias
    std::cout << "Training completed in " << perceptron.get_errors().size() << " epochs." << std::endl;
    std::cout << "Learned weights: [" << perceptron.get_weights()[0] << ", " 
              << perceptron.get_weights()[1] << "]" << std::endl;
    std::cout << "Learned bias: " << perceptron.get_bias() << std::endl << std::endl;
    
    // Test the perceptron
    std::cout << "Testing AND gate:" << std::endl;
    std::cout << "----------------" << std::endl;
    
    std::vector<int> predictions = perceptron.predict(X);
    
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] -> Output: " << predictions[i] 
                  << " (Expected: " << y[i] << ")" << std::endl;
    }
    
    return 0;
}
