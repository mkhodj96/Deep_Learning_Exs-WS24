import numpy as np
from .Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.softmax_output = None # Stores softmax output for backpropagation

    def forward(self, input_tensor):
        # Stabilize input by subtracting the max value in each row (prevent overflow)
        exp_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        # Compute softmax by normalizing exponentiated values
        self.softmax_output = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.softmax_output

    def backward(self, error_tensor):
        # Compute gradient of loss w.r.t. input logits
        gradient_logits = self.softmax_output * error_tensor
        sum_gradients = gradient_logits.sum(axis=1, keepdims=True) # Class-wise gradient sum
        # Adjust for interactions between classes
        gradient_logits -= self.softmax_output * sum_gradients
        return gradient_logits
