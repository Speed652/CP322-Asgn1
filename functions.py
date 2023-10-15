from scipy.spatial.distance import euclidean
import numpy as np

class LogisticRegression:
    def init(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias
        self.theta = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            # Compute predictions
            z = np.dot(X, self.theta) + self.bias
            predictions = self.sigmoid(z)

            # Compute gradients
            dw = (1 / len(X)) * np.dot(X.T, (predictions - y))
            db = (1 / len(X)) * np.sum(predictions - y)

            # Update parameters
            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.theta) + self.bias
        predictions = self.sigmoid(z)
        return np.round(predictions)
    

class KNearestNeighbor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            distances = [euclidean(x, x_train) for x_train in self.X_train]
            sorted_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in sorted_indices]
            predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))

        return np.array(predictions)

def clean_data(data):
    # Remove examples with missing or malformed features
    clean_data = []
    for example in data:
        if not any([val is None or isinstance(val, str) and val.strip() == '' for val in example]):
            clean_data.append(example)

    clean_data = np.array(clean_data)

    # Handle one-hot encoding for categorical variables
    categorical_indices = [1]  # Assuming the second column is categorical

    for idx in categorical_indices:
        unique_values = np.unique(clean_data[:, idx])
        for val in unique_values:
            one_hot_encoded = (clean_data[:, idx] == val).astype(np.int)
            clean_data = np.hstack((clean_data, one_hot_encoded.reshape(-1, 1)))

    # Remove the original categorical column after one-hot encoding
    clean_data = np.delete(clean_data, categorical_indices, axis=1)

    return clean_data