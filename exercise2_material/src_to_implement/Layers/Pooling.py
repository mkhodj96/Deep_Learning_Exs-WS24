import numpy as np
from .Base import BaseLayer
import math


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        # Initialize the pooling layer with the given stride and pooling window sizes.
        # The stride determines the step size for the pooling operation, while the pooling shape
        # determines the size of the pooling window (height and width).

        BaseLayer.__init__(self)
        self.stride_shape_y, self.stride_shape_x = stride_shape[0], stride_shape[1]
        self.pooling_shape_y, self.pooling_shape_x = pooling_shape[0], pooling_shape[1]
        self.image_status = None
        self.input_tensor_shape = None
        self.output_index = []


    def forward(self, input_tensor):
        # Perform the forward pass of the pooling layer (max pooling).
        # The operation reduces the spatial dimensions of the input tensor
        # by taking the maximum value within each pooling window.
        self.input_tensor_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        max_batch = []
        max_batch_index = []
        for img in range(0, batch_size):  # Loop through each image in the batch.
            image = input_tensor[img]
            # Determine if the input is 1D or 2D (spatial dimensions).
            if len(image.shape) == 2:
                self.image_status = '1D'
            elif len(image.shape) == 3:
                self.image_status = '2D'

                # Extract spatial dimensions (height, width) and channels from the input image.
                channel = image.shape[0]
                row = image.shape[1]
                coloumn = image.shape[2]
                max_ch = []
                max_ch_index = []
                for ch in range(0, channel):  # Loop through each channel.
                    max_row = []  # Store pooled results for each row
                    max_index = [] # Store indices of maximum values for each row.
                    for i in range(0, math.floor(row / self.stride_shape_y)):  # Loop over pooling rows.
                        max_col = []  # Store pooled results for each column.
                        for j in range(0, math.floor(coloumn / self.stride_shape_x)):  # Loop over pooling columns.
                            # Define the boundaries of the current pooling window.
                            start_x = j * self.stride_shape_x
                            end_x = start_x + self.pooling_shape_x
                            start_y = i * self.stride_shape_y
                            end_y = start_y + self.pooling_shape_y
                            if end_x <= coloumn and end_y <= row:

                                # Extract the pooling window and compute the maximum value.
                                arr = image[ch][start_y:end_y, start_x:end_x]
                                maximum = np.amax(arr)
                                index = np.where(arr == maximum)

                                # Calculate the position relative to the original image.
                                position_row = int(index[0][0]) + i * self.stride_shape_y
                                position_column = int(index[1][0]) + j * self.stride_shape_x

                                # Store the maximum value and its position.
                                max_index.append([position_row, position_column])
                                max_col.append(maximum)

                            # maximum = np.amax(image[ch][start_y:end_y, start_x:end_x])
                            # max_col.append(maximum)
                        max_col_array = np.array(max_col)
                        # print("max_col_array: ",max_col_array.shape)
                        max_row.append(max_col_array)
                    max_row_array = np.array(max_row)
                    max_index_array = np.array(max_index)
                    max_ch_index.append(max_index_array)
                    # print("max_row_array: ", max_row_array.shape)
                    max_ch.append(max_row_array)
                max_ch_array = np.array(max_ch)
                max_ch_index_array = np.array(max_ch_index)

                max_batch_index.append(max_ch_index_array) # Add channel results to the batch.
                max_batch.append(max_ch_array) # Add channel indices to the batch.

        self.output_index = np.array(max_batch_index)  # Store indices for backpropagation.
        output = np.array(max_batch)  # Return the pooled output tensor.

        return output

    def backward(self, error_tensor):
        # Perform the backward pass of the pooling layer.
        # The gradients are propagated back ony to the positions of the maximum values.
        batch_size = error_tensor.shape[0]  # Number of samples in the batch.
        new_array_batch = [] # To store the gradient for the input tensor.

        for img in range(0, batch_size):  # Loop through each image in the batch.
            error = error_tensor[img]
            channel = error.shape[0]
            new_array_ch = []
            for ch in range(0, channel):  # Loop through each channel.
                new_array = np.zeros((self.input_tensor_shape[-2], self.input_tensor_shape[-1]))
                row_shape = error.shape[-2]
                column_shape = error.shape[-1]
                i = 0
                for row in range(0, row_shape):  # Loop through each row.
                    for col in range(0, column_shape):
                        y = self.output_index[img][ch][i][0]
                        x = self.output_index[img][ch][i][1]
                        i = i + 1
                        new_array[y, x] = new_array[y, x] + error[ch][row][col]
                new_array_ch.append(new_array) # Append reconstructed error for the channel.

            # Append results for the image.
            new_array_batch.append(np.array(new_array_ch))

        error_prev = np.array(new_array_batch) # Combine reconstructed errors for the entire batch.

        return error_prev # Return the error tensor for the previous layer.