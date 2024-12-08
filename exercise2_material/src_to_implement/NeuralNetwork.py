import copy
import numpy as np
from Layers import *
from Optimization import *


class NeuralNetwork(object):
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None, loss_layer=None):
        self.optimizer = optimizer # Optimizer for trainable layers
        self.loss = [] # Track loss during training
        self.layers = [] # List of layers in the network
        self.data_layer = None  # Source of input and label tensors
        self.loss_layer = Loss.CrossEntropyLoss()  # Loss function for the network
        self.input_tensor = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer  # Default is None if not provided
        self.bias_initializer = bias_initializer  # Default is None if not provided
        self.loss_layer = loss_layer if loss_layer else Loss.CrossEntropyLoss()  # Use provided or default loss layer


    def forward(self):
        """
        Perform forward pass through the network to compute loss.

        Returns:
        - loss: Computed loss for the current input and labels.
        """
        self.input_tensor, self.label_tensor = self.data_layer.next()
        current_weights = self.input_tensor
        
        # Pass input through all layers
        for layer in self.layers:
            current_weights = layer.forward(current_weights)  # layer's forward
        # Compute loss using predictions and labels
        predictions = current_weights  # last layers value are predictive
        return self.loss_layer.forward(predictions, self.label_tensor)

    def backward(self):
        """
        Perform backward pass through the network to compute gradients.
        """
        error = self.loss_layer.backward(self.label_tensor)
        for i in range(1, len(self.layers) + 1):
            layer = self.layers[-i]  # iterating layers for back-propagate
            error = layer.backward(error)
        return error

    def append_layer(self, layer):
        """
        Add a layer to the network. If the layer is trainable, assign a copy of the optimizer.
        """
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if self.weights_initializer and self.bias_initializer:
                layer.initialize(self.weights_initializer, self.bias_initializer)  # Initialize trainable layers
        self.layers.append(layer)

    def train(self, iterations):
        """
        Train the network for a specified number of iterations.

        Args:
        - iterations: Number of training iterations.
        """
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        """
        Perform inference on input data.

        Args:
        - input_tensor: Input data for testing.

        Returns:
        - Predictions after forward pass through the network.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
