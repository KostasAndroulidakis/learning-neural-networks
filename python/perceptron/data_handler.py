import numpy as np

def generate_linearly_separable_data(n_samples=100, n_features=2, noise=0.1, random_state=None):
    """
    Generate a synthetic linearly separable dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        noise (float): Amount of noise to add
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is the feature array and y is the target array
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate random weight vector to define decision boundary
    weights = np.random.randn(n_features)
    
    # Generate random data points
    X = np.random.randn(n_samples, n_features)
    
    # Compute raw output: w·x (no bias for simplicity)
    raw_output = np.dot(X, weights)
    
    # Add noise
    if noise > 0:
        raw_output += np.random.normal(0, noise, size=n_samples)
    
    # Create binary target: 1 if w·x > 0, 0 otherwise
    y = np.where(raw_output > 0, 1, 0)
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into training and test sets.
    
    Args:
        X (numpy.ndarray): Feature array
        y (numpy.ndarray): Target array
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize_data(X, X_test=None):
    """
    Normalize features using min-max scaling.
    
    Args:
        X (numpy.ndarray): Training feature array
        X_test (numpy.ndarray, optional): Test feature array to normalize with training stats
        
    Returns:
        numpy.ndarray or tuple: Normalized X or (X_normalized, X_test_normalized)
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    
    # Avoid division by zero
    X_range[X_range == 0] = 1
    
    X_normalized = (X - X_min) / X_range
    
    if X_test is not None:
        X_test_normalized = (X_test - X_min) / X_range
        return X_normalized, X_test_normalized
    
    return X_normalized

def load_iris_binary():
    """
    Load a binary version of the Iris dataset (setosa vs others).
    Demonstrates how to load a real dataset.
    
    Returns:
        tuple: (X, y) where X is the feature array and y is the target array
    """
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data[:, :2]  # Use only the first two features for simplicity
        y = (iris.target == 0).astype(int)  # Setosa (1) vs others (0)
        return X, y
    except ImportError:
        print("scikit-learn not available, returning synthetic data instead")
        return generate_linearly_separable_data(random_state=42)