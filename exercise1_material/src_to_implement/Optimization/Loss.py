import numpy as np


# Cross-entropy loss: Common loss function for classification problems with probability outputs.

class CrossEntropyLoss:
    def __init__(self):
        self.label_tensor = None # Stores ground truth labels
        self.prediction_tensor = None  # Stores predicted probabilities

    def forward(self, prediction_tensor, label_tensor):
        """
        Compute the cross-entropy loss.

        Args:
        - prediction_tensor: Predicted probabilities (e.g., output of softmax).
        - label_tensor: Ground truth labels (one-hot encoded).

        Returns:
        - loss: Scalar cross-entropy loss value.
        """
        self.prediction_tensor = prediction_tensor
        # Avoid log(0) by adding a small epsilon to predictions
        loss = -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        """
        Compute the gradient of the loss with respect to the predictions.

        Args:
        - label_tensor: Ground truth labels.

        Returns:
        - Gradient of the loss with respect to predictions.
        """
        return -(1 / self.prediction_tensor) * label_tensor
