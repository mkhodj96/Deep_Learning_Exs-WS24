import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        # Initialize the Flatten layer, setting it as non-trainable.
        # Store the input tensor shape to restore dimensions during the backward pass.
        super().__init__()
        self.input_shape = None # To save the shape of the input tensor for later use.
        self.trainable = False # Flatten layer does not have trainable parameters.

    def forward(self, input_tensor):
        # Save the shape of the input tensor for use in the backward pass.
        self.input_shape = input_tensor.shape
        batch_size = self.input_shape[0]  # First dimension (batch size) is preserved.
        # Flatten the remaining dimensions into a single vector for each batch.
        # The '-1' automatically infers the size of the flattened dimension based on total elements.
        tensor_flattened = input_tensor.reshape(batch_size, -1)
        return tensor_flattened # Return the flattened tensor for further layers.

    def backward(self, error_tensor):
        # Reshape the error tensor back to the original input shape using the stored input shape.
        return error_tensor.reshape(self.input_shape)
