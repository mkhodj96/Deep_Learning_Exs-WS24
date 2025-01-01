import numpy as np
from Layers import Base, Helpers 
import copy

class BatchNormalization(Base.BaseLayer):
    
    # Initialization of the layer
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self._optimizer = None
        self.moving_mean = None
        self.moving_var = None
        self.decay = 0.8
    
    # Initialize weights and biases
    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)


    # Forward pass of Batch Normalization
    def forward(self, input_tensor):
        X = input_tensor
        conv = False
        if X.ndim == 4: # Check for convolutional input
            conv = True
            X = self.reformat(X)  # Reformat tensor for processing
        self.X = X
        if self.testing_phase: # For testing phase
            if self.moving_mean is None or self.moving_var is None:
                print("please train the model before testing")
            self.mean = self.moving_mean
            self.var = self.moving_var
        else:  # For training phase
            self.mean = np.mean(X, axis= 0)
            self.var = np.var(X, axis=0)
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_var = copy.deepcopy(self.var)
            else:
                self.moving_mean = self.moving_mean * self.decay + self.mean * (1 - self.decay)
                self.moving_var = self.moving_var * self.decay + self.var * (1 - self.decay)
        
        # Normalize and scale/shift input        
        self.X_hat = (X - self.mean) / np.sqrt(self.var + np.finfo(float).eps)
        out = self.gamma * self.X_hat + self.beta
        if conv:
            out = self.reformat(out)   # Reformat back to original shape
        return out


    # Backward pass of Batch Normalization
    def backward(self, error_tensor):
        E = error_tensor
        conv = False
        if E.ndim == 4: # Check for convolutional shape
            conv = True
            conv = True
            E = self.reformat(E)
        # Gradients with respect to weights and biases
        dgamma = np.sum(E * self.X_hat, axis=0)
        dbeta = np.sum(E, axis=0)
        # Gradient with respect to inputs
        grad = Helpers.compute_bn_gradients(E, self.X, self.gamma, self.mean, self.var)
       
        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, dgamma)
            self._optimizer.bias.calculate_update(self.beta, dbeta)

        if conv:
            grad = self.reformat(grad)

        # Cache gradients for weights and biases
        self.gradient_weights = dgamma
        self.gradient_bias = dbeta
        return grad
    

    # Reformat tensor between 4D and 2D shapes
    def reformat(self, tensor):
        if tensor.ndim == 4:  # Reformat 4D tensor to 2D
            self.reformat_shape = tensor.shape
            B, H, M, N = tensor.shape
            tensor = tensor.reshape(B, H, M * N)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(B * M * N, H)
            return tensor
        else: # Reformat 2D tensor back to 4D
            B, H, M, N = self.reformat_shape
            tensor = tensor.reshape(B, M * N, H)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(B, H, M, N)
            return tensor

   
    # Property for weights (gamma)   
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    
    # Property for biases (beta)    
    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta


    # Property for optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)