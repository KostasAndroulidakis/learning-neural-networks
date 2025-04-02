#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include "Perceptron.h"

// Structure to hold evaluation metrics
struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    struct {
        int tp;  // True positives
        int fp;  // False positives
        int tn;  // True negatives
        int fn;  // False negatives
    } confusion_matrix;
};

// Calculate and return model performance metrics
Metrics evaluate_model(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    Metrics metrics;
    
    // Initialize confusion matrix
    metrics.confusion_matrix = {0, 0, 0, 0};
    
    // Calculate confusion matrix
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) metrics.confusion_matrix.tp++;
        if (y_true[i] == 0 && y_pred[i] == 1) metrics.confusion_matrix.fp++;
        if (y_true[i] == 0 && y_pred[i] == 0) metrics.confusion_matrix.tn++;
        if (y_true[i] == 1 && y_pred[i] == 0) metrics.confusion_matrix.fn++;
    }
    
    // Calculate accuracy
    metrics.accuracy = static_cast<double>(metrics.confusion_matrix.tp + metrics.confusion_matrix.tn) / 
                       static_cast<double>(y_true.size());
    
    // Calculate precision (handle division by zero)
    int precision_denominator = metrics.confusion_matrix.tp + metrics.confusion_matrix.fp;
    metrics.precision = (precision_denominator > 0) ? 
                        static_cast<double>(metrics.confusion_matrix.tp) / precision_denominator : 0.0;
    
    // Calculate recall (handle division by zero)
    int recall_denominator = metrics.confusion_matrix.tp + metrics.confusion_matrix.fn;
    metrics.recall = (recall_denominator > 0) ? 
                     static_cast<double>(metrics.confusion_matrix.tp) / recall_denominator : 0.0;
    
    // Calculate F1 score (handle division by zero)
    double precision_recall_sum = metrics.precision + metrics.recall;
    metrics.f1_score = (precision_recall_sum > 0) ? 
                       2.0 * metrics.precision * metrics.recall / precision_recall_sum : 0.0;
    
    return metrics;
}

// Print evaluation metrics
void print_metrics(const Metrics& metrics) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Model Performance Metrics:" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "Accuracy:  " << metrics.accuracy * 100 << "%" << std::endl;
    std::cout << "Precision: " << metrics.precision << std::endl;
    std::cout << "Recall:    " << metrics.recall << std::endl;
    std::cout << "F1 Score:  " << metrics.f1_score << std::endl;
    std::cout << std::endl;
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << "True Positives:  " << metrics.confusion_matrix.tp << std::endl;
    std::cout << "False Positives: " << metrics.confusion_matrix.fp << std::endl;
    std::cout << "True Negatives:  " << metrics.confusion_matrix.tn << std::endl;
    std::cout << "False Negatives: " << metrics.confusion_matrix.fn << std::endl;
}

// Print learning curve (misclassifications per epoch)
void print_learning_curve(const std::vector<int>& errors) {
    std::cout << "Learning Curve (Misclassifications per Epoch):" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    
    // Find max error to scale the chart
    int max_error = *std::max_element(errors.begin(), errors.end());
    int chart_width = 50;  // Width of the chart in characters
    
    for (size_t epoch = 0; epoch < errors.size(); ++epoch) {
        std::cout << "Epoch " << std::setw(3) << epoch + 1 << ": ";
        
        // Print error count
        std::cout << std::setw(3) << errors[epoch] << " ";
        
        // Print a bar chart scaled to max error
        int bar_length = static_cast<int>(static_cast<double>(errors[epoch]) / max_error * chart_width);
        std::cout << std::string(bar_length, '*') << std::endl;
    }
    std::cout << std::endl;
}

// Plot decision boundary (simplified text-based visualization for 2D data)
void plot_decision_boundary(const std::vector<std::vector<double>>& X, 
                           const std::vector<int>& y, 
                           const Perceptron& perceptron) {
    if (X[0].size() != 2) {
        std::cout << "Decision boundary visualization requires 2D data." << std::endl;
        return;
    }
    
    // Determine bounds of the plot
    double x_min = X[0][0], x_max = X[0][0];
    double y_min = X[0][1], y_max = X[0][1];
    
    for (const auto& sample : X) {
        x_min = std::min(x_min, sample[0]);
        x_max = std::max(x_max, sample[0]);
        y_min = std::min(y_min, sample[1]);
        y_max = std::max(y_max, sample[1]);
    }
    
    // Add margin to bounds
    double margin_x = (x_max - x_min) * 0.1;
    double margin_y = (y_max - y_min) * 0.1;
    x_min -= margin_x;
    x_max += margin_x;
    y_min -= margin_y;
    y_max += margin_y;
    
    // Create text-based visualization
    int grid_size = 20;
    std::vector<std::vector<char>> grid(grid_size, std::vector<char>(grid_size, ' '));
    
    // Plot the decision boundary
    const std::vector<double>& weights = perceptron.get_weights();
    double bias = perceptron.get_bias();
    
    // Calculate the decision boundary line: w[0]*x + w[1]*y + bias = 0
    // Rearranged as y = -(w[0]*x + bias) / w[1]
    if (weights[1] != 0) {  // Avoid division by zero
        for (int i = 0; i < grid_size; ++i) {
            double x = x_min + (x_max - x_min) * i / (grid_size - 1);
            double y = -(weights[0] * x + bias) / weights[1];
            
            // Convert to grid coordinates
            int grid_y = static_cast<int>((y - y_min) / (y_max - y_min) * (grid_size - 1));
            
            // Ensure grid_y is within bounds
            if (grid_y >= 0 && grid_y < grid_size) {
                grid[grid_size - 1 - grid_y][i] = '-';
            }
        }
    }
    
    // Plot the data points
    for (size_t i = 0; i < X.size(); ++i) {
        int grid_x = static_cast<int>((X[i][0] - x_min) / (x_max - x_min) * (grid_size - 1));
        int grid_y = static_cast<int>((X[i][1] - y_min) / (y_max - y_min) * (grid_size - 1));
        
        // Ensure grid coordinates are within bounds
        if (grid_x >= 0 && grid_x < grid_size && grid_y >= 0 && grid_y < grid_size) {
            grid[grid_size - 1 - grid_y][grid_x] = (y[i] == 1) ? '+' : 'o';
        }
    }
    
    // Print the visualization
    std::cout << "Decision Boundary Visualization:" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "+ : Class 1, o : Class 0, - : Decision Boundary" << std::endl << std::endl;
    
    for (const auto& row : grid) {
        for (char c : row) {
            std::cout << c << ' ';
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "X-axis: " << x_min << " to " << x_max << std::endl;
    std::cout << "Y-axis: " << y_min << " to " << y_max << std::endl;
    std::cout << std::endl;
}

#endif // UTILS_H
