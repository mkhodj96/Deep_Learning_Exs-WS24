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

class SgdWithMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        # Initialize the learning rate and momentum factor.
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None # Used to store the moving average of gradients.

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Initialize velocity with zeros if it hasn't been initialized yet.
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        # Update velocity using the momentum formula: v = m * v - lr * gradient.
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_tensor
        # Update weights: w = w + v (apply the velocity to the weights).
        return weight_tensor + self.velocity


class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        # Initialize hyperparameters: learning rate, mu (beta1), and rho (beta2).
        self.learning_rate = learning_rate
        self.mu = mu  # Exponential decay rate for the first moment (mean of gradients).
        self.rho = rho # Exponential decay rate for the second moment (variance of gradients).
        self.epsilon = 1e-8  # small constant introduced for numerical stability
        self.m = None  # First moment vector (mean of gradients).
        self.v = None  # Second moment vector (mean of squared gradients).
        self.t = 0  # Time step (iteration count).

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Initialize first and second moment vectors with zeros if they haven't been set yet.
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)  # Same shape as weights.
            self.v = np.zeros_like(weight_tensor)
        self.t += 1 # Increment the time step.
        # Update biased first moment: m = beta1 * m + (1 - beta1) * gradient.
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        # Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2.
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)
        # Compute bias-corrected first moment: m_hat = m / (1 - beta1^t).
        m_hat = self.m / (1 - self.mu ** self.t)
        # Compute bias-corrected second moment: v_hat = v / (1 - beta2^t).
        v_hat = self.v / (1 - self.rho ** self.t)
        # Update weights using Adam formula: w = w - lr * m_hat / (sqrt(v_hat) + epsilon).
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
