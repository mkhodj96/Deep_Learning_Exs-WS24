import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        # Ensure that the resolution is evenly divisible by twice the tile size
        assert resolution % (2 * tile_size) == 0, "Resolution must be evenly dividable by 2 * tile_size."
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # Create column and row index arrays for the image dimensions
        # These arrays represent the pixel coordinates in the checkerboard pattern
        grid_x = np.arange(self.resolution)[:, None]  # Column vector of indices (0 to resolution-1)
        grid_y = np.arange(self.resolution)[None, :]  # Row vector of indices (0 to resolution-1)

        # Convert pixel coordinates to tile indices by dividing by `tile_size`
        # This creates tile coordinates, where each tile spans `tile_size` pixels
        grid_x = grid_x // self.tile_size  # Each value tells which tile row a pixel belongs to
        grid_y = grid_y // self.tile_size  # Each value tells which tile column a pixel belongs to

        # Determine the color of each tile based on its position:
        # - Tiles with the same row and column parity (both even or both odd) are black (0)
        # - Tiles with differing parity are white (1)
        self.output = np.where((grid_x % 2) == (grid_y % 2), 0, 1)  # Create the checkerboard pattern

        # np.where: used for selecting elements from two arrays based on the condition
        # If parity is same, insert black tile and vice versa

        # Caller receives copy of the array rather than a reference to the original array stored within the object
        return self.output.copy()

    def show(self):
        # Display the checkerboard pattern if it exists
        if self.output is None:
            print("The pattern does not exist.")
        else:
        # Show the checkerboard image in grayscale, with the axis labels hidden
            plt.imshow(self.output, cmap='gray')
            plt.axis('off')
            plt.show()


class Spectrum:
    def __init__(self, resolution):
        # Ensure the resolution is an integer
        assert isinstance(resolution, int), "The resolution must be an integer."
        self.resolution = resolution
        self.output = None

    def draw(self):
        # Initialize a 3D array to store RGB color values, with dimensions (resolution, resolution, 3)
        # and data type as float64 for color gradients
        self.output = np.zeros((self.resolution, self.resolution, 3), dtype=np.float64)
        
        # Generate a horizontal gradient for the red channel (from 0 to 1 across the width)
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)  # Red channel 

        # Generate a horizontal gradient for the blue channel (from 1 to 0 across the width, reversing red)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)  # Blue channel 

        # Generate a vertical gradient for the green channel (from 0 to 1 down the height)
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)  # Green channel
        return self.output.copy()

    def show(self):
        # Check if the color pattern is generated, and display it if so
        if self.output is None:
            print("The pattern does not exist.")
        else:
        # Display the color spectrum image without axis labels
            plt.imshow(self.output)
            plt.axis('off')
            plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        # Generate arrays for x and y coordinates from 0 up to resolution - 1
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        # Create a 2D grid of x and y coordinates
        x, y = np.meshgrid(x, y)

        # Calculate distances from each point to the center of the circle, distances = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        distances = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)

        # Generate an array where points inside the circle are 1, and points outside are 0
        self.output = np.where(distances <= self.radius, 1, 0)

        return self.output.copy()

    def show(self):
        # Display the generated circle image in binary colors (black and white)

        plt.imshow(self.output, cmap='binary')
        plt.axis('off') # Hide the axis
        plt.show()
