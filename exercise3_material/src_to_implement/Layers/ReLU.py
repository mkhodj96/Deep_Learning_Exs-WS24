import numpy as np
from .Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None # Stores the input for backpropagation

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor) # Apply ReLU activation (element-wise max)

    def backward(self, error_tensor):
        # Create a mask: 1 where input > 0, otherwise 0
        grad_input = (self.input_tensor > 0).astype(float)
        return grad_input * error_tensor  # Element-wise product with error tensor
