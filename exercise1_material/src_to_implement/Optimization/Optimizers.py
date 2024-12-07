import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        # assert isinstance(learning_rate, float), "Learning rate must be a float."
        
        # Initialize the SGD optimizer with a specified learning rate
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculate the updated weights using the Stochastic Gradient Descent (SGD) algorithm.

        Args:
        - weight_tensor: The current weights of the model (numpy array or similar).
        - gradient_tensor: The computed gradients for the weights (same shape as weight_tensor).

        Returns:
        - updated_weights: The adjusted weights after applying the SGD update rule.
        """
        
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor

        return updated_weights
