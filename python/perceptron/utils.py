import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, perceptron):
    """Plot the decision boundary and data points"""
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions for each point in the mesh
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    
    # Plot the decision boundary
    slope = -perceptron.weights[0] / perceptron.weights[1]
    intercept = -perceptron.bias / perceptron.weights[1]
    x_boundary = np.linspace(x_min, x_max, 100)
    y_boundary = slope * x_boundary + intercept
    plt.plot(x_boundary, y_boundary, 'r-', linewidth=2)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')

def plot_learning_curve(errors):
    """Plot the learning curve showing misclassifications per epoch"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Misclassifications')
    plt.title('Perceptron Learning Curve')
    plt.grid(True)
    return plt
    
def evaluate_model(y_true, y_pred):
    """Calculate and return model performance metrics"""
    accuracy = np.mean(y_pred == y_true)
    
    # Calculate confusion matrix elements
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Store in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'tp': tp, 'fp': fp,
            'tn': tn, 'fn': fn
        }
    }
    
    return metrics