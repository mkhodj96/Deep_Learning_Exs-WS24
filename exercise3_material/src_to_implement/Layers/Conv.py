import copy
import math
import numpy as np
import scipy.signal

from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.input_tensor = None
        self.trainable = True # Indicates that this layer has trainable parameters.

        self.num_kernels = num_kernels # Number of kernels (filters) in the convolution layer.
        self.stride_shape = stride_shape # Defines the stride size for spatial dimensions.
        self.convolution_shape = convolution_shape   # Shape of each kernel (channels, spatial dimensions).

        # Random initialization for weights (kernels) and bias.
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.random(num_kernels)

        self._padding = "same" # Default padding: ensures output has the same spatial dimensions as input.
        self._optimizer = None # Placeholder for the optimizer object.
        self._gradient_weights = None # Gradient for weights (computed during backprop).
        self._gradient_bias = None  # Gradient for bias (computed during backprop).

    def forward(self, input_tensor):
        # Store the input tensor for backpropagation.
        batch_size = input_tensor.shape[0]# Extract batch size.

        # Define the output tensor dimensions based on input shape and stride.
        if len(input_tensor.shape) == 3:  # Case: 1D convolution.

            hin = input_tensor.shape[2]
            hout = math.ceil(hin / self.stride_shape[0])
            out = np.zeros((batch_size, self.num_kernels, hout))
        elif len(input_tensor.shape) == 4:  # Case: 2D convolution.
            # For 2D convolution
            hin = input_tensor.shape[2]
            win = input_tensor.shape[3]
            hout = math.ceil(hin / self.stride_shape[0])
            wout = math.ceil(win / self.stride_shape[1])
            out = np.zeros((batch_size, self.num_kernels, hout, wout))

        self.input_tensor = input_tensor  # storing for backprop

        # Perform convolution operation for each item in the batch.
        for item in range(batch_size):
            for ker in range(self.num_kernels):
                # Compute correlation between input tensor and kernel weights.
                output = scipy.signal.correlate(input_tensor[item], self.weights[ker],
                                                self._padding)  # Same padding: output size matches input size (stride = 1).
                output = output[output.shape[0] // 2]  # Handle valid padding in channel dimensions.
                # Apply strides: subsample the output tensor.
                if len(self.stride_shape) == 1:
                    output = output[::self.stride_shape[0]]  # Apply stride for 1D.
                elif len(self.stride_shape) == 2:
                    output = output[::self.stride_shape[0], ::self.stride_shape[1]]  # Apply stride for 2D.
                out[item, ker] = output + self.bias[ker]  # Add bias to the output tensor.

        return out

    def backward(self, errT):
        # Backpropagation: calculate gradients for weights and input tensor.
        batch_size = np.shape(errT)[0]
        num_channels = self.convolution_shape[0]

        # Prepare weights for backpropagation (swap axes and flip dimensions for convolution).
        weights = np.swapaxes(self.weights, 0, 1)
        weights = np.fliplr(weights)
        # Initialize gradient tensors.
        error_per_item = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        dX = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:]))
        # Perform convolution for each item in the batch.
        for item in range(batch_size):
            for ch in range(num_channels):
                if len(self.stride_shape) == 1:
                    error_per_item[:, :, ::self.stride_shape[0]] = errT[item]
                else:
                    error_per_item[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = errT[item]
                # Compute gradients with respect to input.
                output = scipy.signal.convolve(error_per_item[item], weights[ch], 'same')
                output = output[output.shape[0] // 2]
                dX[item, ch] = output

        # Compute gradients for weights and biases.
        self._gradient_weights, self._gradient_bias = self.get_weights_biases_gradient(errT)
        # Update weights and biases if an optimizer is set.
        if self.optimizer is not None:
            # Update weights and bias using optimizer
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self._gradient_weights)
            self.bias = copy.deepcopy(self.optimizer).calculate_update(self.bias, self._gradient_bias)

        return dX

    def get_weights_biases_gradient(self, errT):
        # Compute gradients for weights and biases using error tensor.

        global dB
        batch_size = np.shape(errT)[0]
        num_channels = self.convolution_shape[0]
        dW = np.zeros((self.num_kernels, *self.convolution_shape))
        # Initialize error tensor for backpropagation.
        error_per_item = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        for item in range(batch_size):
            if len(self.stride_shape) == 1:
                error_per_item[:, :, ::self.stride_shape[0]] = errT[item]
                dB = np.sum(errT, axis=(0, 2))
                padding_width = ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2))
            else:
                error_per_item[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = errT[item]
                dB = np.sum(errT, axis=(0, 2, 3))
                padding_width = ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),
                                 (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2))

            # Pad the input tensor to compute valid convolution.
            paded_X = np.pad(self.input_tensor[item], padding_width, mode='constant', constant_values=0)
            tmp = np.zeros((self.num_kernels, *self.convolution_shape))
            for ker in range(self.num_kernels):
                for ch in range(num_channels):
                    tmp[ker, ch] = scipy.signal.correlate(paded_X[ch], error_per_item[item][ker], 'valid')
            dW += tmp
        return dW, dB

    @property
    def gradient_weights(self):
        # Getter for weight gradients.
        return self._gradient_weights

    @property
    def gradient_bias(self):
        # Getter for bias gradients.
        return self._gradient_bias

    @property
    def optimizer(self):
        # Getter for the optimizer.
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        # Setter for the optimizer.
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        # Initialize weights and biases using the given initializers.
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)
