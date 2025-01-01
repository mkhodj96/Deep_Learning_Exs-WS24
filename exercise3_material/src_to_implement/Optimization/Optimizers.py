import numpy as np

# Ex03 a parrent class for all the optimizers
class Optimizer(object): 
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()  # inherit from this \base-optimizer".
        # Initialize the SGD optimizer with a specified learning rate
        self.learning_rate = learning_rate
        self.regularizer = None # Optional regularizer for weight updates.  /EX03

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculate the updated weights using the Stochastic Gradient Descent (SGD) algorithm.

        Args:
        - weight_tensor: The current weights of the model (numpy array or similar).
        - gradient_tensor: The computed gradients for the weights (same shape as weight_tensor).

        Returns:
        - updated_weights: The adjusted weights after applying the SGD update rule.
        """
        
        # Copy weight tensor to avoid altering the original data  /EX03
        temp_weights = weight_tensor.copy() if type(weight_tensor) is np.ndarray else weight_tensor

        # Perform the basic SGD weight update
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor

        # Apply regularization /EX03
        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(temp_weights)
    
        return updated_weights

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__()
        # Initialize the learning rate and momentum factor.
        self. learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None # Used to store the moving average of gradients.
        self.regularizer = None  # Optional regularizer for weight updates.  /EX03


    def calculate_update(self, weight_tensor, gradient_tensor):
        # Copy weight tensor to keep the original data /EX03
        temp_weights = weight_tensor.copy() if type(weight_tensor) is np.ndarray else weight_tensor

        # Initialize velocity with zeros if not already initialized
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        # Compute velocity update using momentum and learning rate
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_tensor

        # Update weights using the velocity
        weight_tensor = weight_tensor + self.velocity

        # Apply regularization /EX03
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(temp_weights)
        
        return weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        super().__init__()

        # Initialize hyperparameters: learning rate, mu (beta1), and rho (beta2).
        self.learning_rate = learning_rate
        self.mu = mu  # Exponential decay rate for the first moment (mean of gradients).
        self.rho = rho # Exponential decay rate for the second moment (variance of gradients).
        self.epsilon = 1e-8  # small constant introduced for numerical stability
        self.m = None  # First moment vector (mean of gradients).
        self.v = None  # Second moment vector (mean of squared gradients).
        self.t = 0  # Time step (iteration count).
        self.regularizer = None  # Optional regularizer for weight updates.  /EX03

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Copy weight tensor to avoid altering the original data /EX03
        temp_weights = weight_tensor.copy() if type(weight_tensor) is np.ndarray else weight_tensor

        # Initialize first and second moment vectors with zeros if not already set
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)

        # Increment the time step
        self.t += 1

        # Update biased first moment (mean of gradients)
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor

        # Update biased second moment (mean of squared gradients)
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        # Compute bias-corrected first moment
        m_hat = self.m / (1 - self.mu ** self.t)

        # Compute bias-corrected second moment
        v_hat = self.v / (1 - self.rho ** self.t)

        # Update weights using Adam formula
        weight_tensor = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Apply regularization /EX03
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(temp_weights)

        return weight_tensor
        

        