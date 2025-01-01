import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Fully connected layers for hidden state and output computation
        self.FC_h = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_y = FullyConnected(hidden_size, output_size)
        
        # Initialize placeholders for gradients and weights
        self.gradient_weights_n = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights_y = None
        self.weights_h = None
        self.weights = self.FC_h.weights
        self.tan_h = TanH() # Activation function
        self.bptt = 0
        self.h_t = None
        self.prev_h_t = None
        self.batch_size = None
        self.optimizer = None
        self.h_memory = []

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        
        # Initialize hidden state
        if self._memorize:
            if self.h_t is None:
                self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))
            else:
                self.h_t[0] = self.prev_h_t
        else:
            self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))

        output_tensor = np.zeros((self.batch_size, self.output_size))

        # Process each time step in the batch
        for i in range(self.batch_size):
            hidden_ax = self.h_t[i][np.newaxis, :]
            input_ax = input_tensor[i][np.newaxis, :]
            input_new = np.concatenate((hidden_ax, input_ax), axis = 1)

            self.h_memory.append(input_new)

            w_t = self.FC_h.forward(input_new)
            input_new = np.concatenate((np.expand_dims(self.h_t[i], 0), np.expand_dims(input_tensor[i], 0)), axis=1)
            self.h_t[i+1] = TanH().forward(w_t) 
            output_tensor[i] = (self.FC_y.forward(self.h_t[i + 1][np.newaxis, :]))
        
        self.prev_h_t = self.h_t[-1]
        self.input_tensor = input_tensor

        return output_tensor

    def backward(self, error_tensor):

        self.out_error = np.zeros((self.batch_size, self.input_size))

        # Gradient placeholders for weights
        self.gradient_weights_y = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights_h = np.zeros((self.hidden_size+self.input_size+1, self.hidden_size))

        count = 0

        grad_tanh = 1-self.h_t[1::] ** 2
        hidden_error = np.zeros((1, self.hidden_size))

        # Backpropagate through time steps
        for i in reversed(range(self.batch_size)):
            yh_error = self.FC_y.backward(error_tensor[i][np.newaxis, :])
            self.FC_y.input_tensor = np.hstack((self.h_t[i+1], 1))[np.newaxis, :]

            grad_yh = hidden_error + yh_error
            grad_hidden = grad_tanh[i]*grad_yh
            xh_error = self.FC_h.backward(grad_hidden)
            hidden_error = xh_error[:, 0:self.hidden_size]
            x_error = xh_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]
            self.out_error[i] = x_error

            con = np.hstack((self.h_t[i], self.input_tensor[i],1))
            self.FC_h.input_tensor = con[np.newaxis, :]
            # Save weights and gradients for optimization
            if count <= self.bptt:
                self.weights_y = self.FC_y.weights
                self.weights_h = self.FC_h.weights
                self.gradient_weights_y = self.FC_y.gradient_weights
                self.gradient_weights_h = self.FC_h.gradient_weights
            count += 1

        # Update weights using optimizer

        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
            self.weights_h = self.optimizer.calculate_update(self.weights_h, self.gradient_weights_h)
            self.FC_y.weights = self.weights_y
            self.FC_h.weights = self.weights_h
        return self.out_error

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_y.initialize(weights_initializer, bias_initializer)
        self.FC_h.initialize(weights_initializer, bias_initializer)

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_n

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_y.gradient_weights = gradient_weights