#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace activation {

// Step activation function: 1 if x ≥ 0, 0 otherwise
inline std::vector<double> step(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] >= 0 ? 1.0 : 0.0;
    }
    return result;
}

// Derivative of step function (technically undefined at 0, but we return 0)
inline std::vector<double> step_derivative(const std::vector<double>& x) {
    return std::vector<double>(x.size(), 0.0);
}

// Sigmoid activation function: 1/(1+e^(-x))
inline std::vector<double> sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        // Clip to avoid overflow
        double x_safe = std::max(-500.0, std::min(x[i], 500.0));
        result[i] = 1.0 / (1.0 + std::exp(-x_safe));
    }
    return result;
}

// Derivative of sigmoid function: sigmoid(x) * (1 - sigmoid(x))
inline std::vector<double> sigmoid_derivative(const std::vector<double>& x) {
    std::vector<double> sig = sigmoid(x);
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sig[i] * (1.0 - sig[i]);
    }
    return result;
}

// ReLU (Rectified Linear Unit) activation function: max(0, x)
inline std::vector<double> relu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::max(0.0, x[i]);
    }
    return result;
}

// Derivative of ReLU function: 1 if x > 0, 0 otherwise
inline std::vector<double> relu_derivative(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0 ? 1.0 : 0.0;
    }
    return result;
}

// Hyperbolic tangent activation function
inline std::vector<double> tanh(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::tanh(x[i]);
    }
    return result;
}

// Derivative of tanh function: 1 - tanh^2(x)
inline std::vector<double> tanh_derivative(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double tanh_x = std::tanh(x[i]);
        result[i] = 1.0 - tanh_x * tanh_x;
    }
    return result;
}

// Single element versions of the functions
inline double step(double x) {
    return x >= 0 ? 1.0 : 0.0;
}

inline double step_derivative(double x) {
    return 0.0;
}

inline double sigmoid(double x) {
    double x_safe = std::max(-500.0, std::min(x, 500.0));
    return 1.0 / (1.0 + std::exp(-x_safe));
}

inline double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

inline double relu(double x) {
    return std::max(0.0, x);
}

inline double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

inline double tanh_derivative(double x) {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

} // namespace activation

#endif // ACTIVATION_H
