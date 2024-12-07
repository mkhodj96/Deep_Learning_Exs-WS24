import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gradient_biases = None # Gradient for biases
        self.gradient_weights = None  # Gradient for weights
        self.input_tensor = None # Stores the input tensor for backward computation
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True # Layer parameters are trainable

        # Initialize weights with uniform random values, including space for the bias
        self.weights = np.random.uniform(0, 1, (output_size, input_size + 1))
        self._optimizer = None # Placeholder for the optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bias_term = np.ones((input_tensor.shape[0], 1))  # Bias vector
        augmented_input = np.hstack([input_tensor, bias_term])  # Add bias to input
        return np.dot(augmented_input, self.weights.T)  # Compute output

    def backward(self, error_tensor):
        bias_term = np.ones((self.input_tensor.shape[0], 1))  # Bias vector
        augmented_input = np.hstack([self.input_tensor, bias_term]) # Add bias to input
        self.gradient_weights = np.dot(error_tensor.T, augmented_input) # Compute weight gradients
        
        # Update weights if an optimizer is set
        if self._optimizer:  
            sgd = self.optimizer
            self.weights = sgd.calculate_update(self.weights, self.gradient_weights)
        # Return gradient with respect to input (excluding bias)
        return np.dot(error_tensor, self.weights[:,:-1])

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
