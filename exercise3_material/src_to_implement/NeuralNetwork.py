import copy
import numpy as np
from Layers import *
from Optimization import *

class NeuralNetwork(object):
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None, loss_layer=None):
        self.optimizer = optimizer # Optimizer for trainable layers
        self.loss = [] # Track loss during training
        self.layers = [] # List of layers in the network
        self.data_layer = None  # Source of input and label tenso,rs
        self.loss_layer = Loss.CrossEntropyLoss()  # Loss function for the network
        self.input_tensor = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer  # Default is None if not provided
        self.bias_initializer = bias_initializer  # Default is None if not provided
        self.loss_layer = loss_layer if loss_layer else Loss.CrossEntropyLoss()  # Use provided or default loss layer
        self._phase = None #setting the phase of each of its layers accordingly


    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
        # Propagate phase to all layers
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = (phase == 'test')


    def forward(self):
        """
        Perform forward pass through the network to compubujl-te loss.

        Returns:
        - loss: Computed loss for the current input and labels.

        Added after EX03: logic to compute regularization loss using the regularizer from the optimizer. 
        Summed it with the global loss.
        """

        self.input_tensor, self.label_tensor = self.data_layer.next()
        current_tensor = self.input_tensor

        reg_loss = 0  # Track regularization loss
        # Pass input through all layers
        for layer in self.layers:
            current_tensor = layer.forward(current_tensor)
            if self.optimizer.regularizer is not None and hasattr(layer, 'weights'):
                reg_loss += self.optimizer.regularizer.norm(layer.weights)

        # Compute loss using predictions and labels
        predictions = current_tensor
        glob_loss = self.loss_layer.forward(predictions, self.label_tensor)

        return glob_loss + reg_loss

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
        Edit after EX03: weights and biases are initialized before assigning the optimizer in append_layer
        """

        if layer.trainable:
            # Initialize weights and biases before assigning optimizer
            if self.weights_initializer and self.bias_initializer:
                layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)


    def train(self, iterations):
        self.phase = 'train'

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
        self.phase = 'test'

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
