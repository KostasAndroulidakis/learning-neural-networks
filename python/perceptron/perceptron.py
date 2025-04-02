import numpy as np


class Perceptron:
    """
    A simple Perceptron implementation.
    
    The Perceptron is a linear binary classifier that learns weights 
    to separate data points into two classes.
    """
    
    def __init__(self, n_inputs, learning_rate=0.01, max_epochs=1000, random_state=None):
        """
        Initialize the Perceptron.
        
        Args:
            n_inputs (int): Number of input features
            learning_rate (float): Learning rate (eta)
            max_epochs (int): Maximum number of training epochs
            random_state (int): Seed for random number generator
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Set random seed for reproducibility
        if random_state:
            np.random.seed(random_state)
            
        # Initialize weights and bias
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        
        # Training history
        self.errors = []
        
    def predict(self, X):
        """
        Make a prediction using the current model.
        
        Args:
            X (numpy.ndarray): Input features, shape (n_samples, n_inputs)
            
        Returns:
            numpy.ndarray: Predicted class labels (0 or 1)
        """
        # Calculate net input: w·x + b
        net_input = np.dot(X, self.weights) + self.bias
        
        # Apply step activation function
        return np.where(net_input >= 0, 1, 0)
    
    def _update_weights(self, X, y, y_pred):
        """
        Update weights based on prediction errors.
        
        Args:
            X (numpy.ndarray): Input features, shape (n_samples, n_inputs)
            y (numpy.ndarray): Target values, shape (n_samples,)
            y_pred (numpy.ndarray): Predicted values, shape (n_samples,)
        """
        # Calculate error
        errors = y - y_pred
        
        # Update weights using the perceptron learning rule
        # Δw = η * (y - ŷ) * x
        delta_w = self.learning_rate * np.dot(errors, X)
        delta_b = self.learning_rate * np.sum(errors)
        
        self.weights += delta_w
        self.bias += delta_b
        
        return np.sum(errors != 0)  # Return number of misclassifications
    
    def fit(self, X, y):
        """
        Train the perceptron on the given data.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_inputs)
            y (numpy.ndarray): Target values (0 or 1), shape (n_samples,)
            
        Returns:
            self: The trained model
        """
        self.errors = []
        
        # Training loop
        for epoch in range(self.max_epochs):
            # Make predictions
            y_pred = self.predict(X)
            
            # Update weights and count errors
            misclassified = self._update_weights(X, y, y_pred)
            self.errors.append(misclassified)
            
            # Stop if all samples are correctly classified
            if misclassified == 0:
                break
                
        return self