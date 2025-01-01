import numpy as np


class BaseLayer(object):
    def __init__(self):

        self.trainable = False # Indicates whether the layer's parameters are trainable
        self.weights = []

        #self.weights = None # Initialized with random values when needed for optimization
        self.testing_phase=False