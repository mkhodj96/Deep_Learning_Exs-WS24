import copy
import numpy as np
from Layers import *
from Optimization import *


class NeuralNetwork(object):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = Loss.CrossEntropyLoss()  # Initialize the loss layer once
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        # iterate through layers, pass input_tensor to each one until loss is generated
        # only input goes in for the first layer
        current_weights = self.input_tensor
        for layer in self.layers:
            current_weights = layer.forward(current_weights)  # layer's forward
            # result gets passed to next layer for forward in successive iterations
        predictions = current_weights  # last layers value are predictive
        return self.loss_layer.forward(predictions, self.label_tensor)

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for i in range(1, len(self.layers) + 1):
            layer = self.layers[-i]  # iterating layers for back-propagate
            error = layer.backward(error)
        return error

    def append_layer(self, layer):
        if layer.trainable:  # if layer parameters need to be updated, ensure there is no shared state between
            # different layer's optimizers
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
