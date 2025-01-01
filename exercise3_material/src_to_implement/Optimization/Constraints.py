import numpy as np


# L2 Regularization
class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha # Regularization strength

    def norm(self, weight_tensor):
        # L2 norm: alpha * sum of squared weights
        return self.alpha * np.sum(np.square(weight_tensor))
    
    def calculate_gradient(self, weight_tensor):
         # Gradient of L2 norm: alpha * weights
        return self.alpha * weight_tensor

# L1 Regularization
class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha # Regularization strength

    def norm(self, weight_tensor):
        # L1 norm: alpha * sum of absolute weights
        return self.alpha * np.sum(np.abs(weight_tensor))
    
    def calculate_gradient(self, weight_tensor):
        # Gradient of L1 norm: alpha * sign of weights
        return np.sign(weight_tensor) * self.alpha
