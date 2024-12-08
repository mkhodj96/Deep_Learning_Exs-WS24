import numpy as np


class Constant:
    def __init__(self, value=0.1):
        self.value = value
    # Initializes a tensor with a constant value for all elements.

    def initialize(self, shape, fan_in=None, fan_out=None):
        return np.full(shape, self.value)


class UniformRandom:
    @staticmethod
    # Initializes a tensor with values drawn from a uniform distribution [0, 1).
    def initialize(shape, fan_in=None, fan_out=None):
        return np.random.uniform(0, 1, size=shape)


class Xavier:
    @staticmethod
    # Implements Xavier initialization: weights are sampled from a normal distribution 
    # with a standard deviation based on the fan-in and fan-out values for balanced gradients.
    def initialize(shape, fan_in, fan_out):
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)


class He:
    @staticmethod
    # Implements He initialization, optimized for ReLU activations. 
    # Weights are sampled from a normal distribution with stddev based on fan-in.
    def initialize(shape, fan_in, fan_out=None):
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, size=shape)
