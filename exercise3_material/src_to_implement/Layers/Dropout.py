import numpy as np
from Layers import Base

# Dropout layer for regularization
class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.prob = probability  # Dropout probability

    def forward(self, input_tensor):
        # pass inputs unchanged during testing
        if self.testing_phase:
            self.mask = np.ones(input_tensor.shape)
        else:
        # Apply mask during training
            temp = np.random.rand(*input_tensor.shape)
            self.mask = (temp < self.prob).astype(float) # Create dropout mask
            self.mask /= self.prob # Scale to maintain expected value
        return input_tensor * self.mask
    
    def backward(self, error_tensor):
        # Backpropagate through the dropout mask
        return error_tensor * self.mask
        