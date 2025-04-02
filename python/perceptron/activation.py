import numpy as np

def step(x):
    """
    Step activation function: 1 if x ≥ 0, 0 otherwise.
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Activated values (0 or 1)
    """
    return np.where(x >= 0, 1, 0)

def step_derivative(x):
    """
    Derivative of step function (technically undefined at 0, 
    but we return 0 for computational purposes).
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Zeros array of same shape as input
    """
    return np.zeros_like(x)

def sigmoid(x):
    """
    Sigmoid activation function: 1/(1+e^(-x))
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Activated values in range (0, 1)
    """
    # Clip to avoid overflow
    x_safe = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_safe))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function: sigmoid(x) * (1 - sigmoid(x))
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Derivative values
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function: max(0, x)
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Activated values
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU function: 1 if x > 0, 0 otherwise
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Derivative values (0 or 1)
    """
    return np.where(x > 0, 1, 0)

def tanh(x):
    """
    Hyperbolic tangent activation function
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Activated values in range (-1, 1)
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of tanh function: 1 - tanh^2(x)
    
    Args:
        x (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Derivative values
    """
    return 1 - np.tanh(x)**2