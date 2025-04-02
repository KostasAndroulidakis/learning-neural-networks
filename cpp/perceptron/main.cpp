#include <iostream>
#include <iomanip>
#include "Perceptron.h"
#include "data_handler.h"
#include "utils.h"
#include "activation.h"

int main() {
    // Set random seed for reproducibility
    int random_seed = 42;
    
    std::cout << "=== Perceptron Implementation in C++ ===" << std::endl << std::endl;
    
    // Generate synthetic dataset
    std::cout << "Generating linearly separable dataset..." << std::endl;
    Dataset dataset = generate_linearly_separable_data(100, 2, 0.1, random_seed);
    
    // Split into training and test sets
    std::cout << "Splitting data into training and test sets..." << std::endl;
    TrainTestSplit split = train_test_split(dataset.X, dataset.y, 0.3, random_seed);
    
    // Normalize features
    std::cout << "Normalizing features..." << std::endl;
    auto normalized = normalize_data(split.X_train, split.X_test);
    split.X_train = normalized.first;
    split.X_test = normalized.second;
    
    // Initialize and train the perceptron
    std::cout << "Initializing perceptron..." << std::endl;
    Perceptron perceptron(2, 0.01, 100, random_seed);
    
    std::cout << "Training perceptron..." << std::endl;
    perceptron.fit(split.X_train, split.y_train);
    
    // Display training results
    const std::vector<int>& errors = perceptron.get_errors();
    std::cout << "Training completed in " << errors.size() << " epochs." << std::endl;
    std::cout << "Final weights: [";
    for (size_t i = 0; i < perceptron.get_weights().size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << perceptron.get_weights()[i];
        if (i < perceptron.get_weights().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Final bias: " << std::fixed << std::setprecision(4) << perceptron.get_bias() << std::endl << std::endl;
    
    // Make predictions on test set
    std::cout << "Making predictions on test set..." << std::endl;
    std::vector<int> y_pred = perceptron.predict(split.X_test);
    
    // Evaluate model performance
    std::cout << "Evaluating model performance..." << std::endl << std::endl;
    Metrics metrics = evaluate_model(split.y_test, y_pred);
    print_metrics(metrics);
    
    // Print learning curve
    print_learning_curve(errors);
    
    // Visualize decision boundary
    plot_decision_boundary(split.X_train, split.y_train, perceptron);
    
    return 0;
}
