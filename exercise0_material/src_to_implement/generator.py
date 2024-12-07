import os.path
import json
import scipy.misc
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # self.mirror_axis=None

        if mirroring:
            self.mirror_axis = [0, 1]  # Set this to valid axis indices for mirroring

        # Validate and store input parameters
        assert (isinstance(file_path, str)), "The file_path must be a string."
        self.file_path = file_path
        assert (isinstance(label_path, str)), "The label_path must be a string."
        self.label_path = label_path
        assert (isinstance(batch_size, int)), "The batch_size must be an integer."
        self.batch_size = batch_size
        assert isinstance(image_size, list) and all(isinstance(item, int) for item in image_size), ("The image_size "
                                                                                                    "must be a list "
        # Dictionary for mapping label indices to class names
                                                                         "of integers.")
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.class_dict = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

        self.current_epoch_no = 0
        self.current_batch = 0

        # Load labels from JSON file
        label_json = open(self.label_path, "rb")  # read, binary
        self.labels = json.load(label_json)  # convert json data into python object
        label_json.close()

        # Load image filenames and remove file extensions
        image_files = os.listdir(self.file_path)  # list all entries in a specified directory
        self.image_names = [img_file[:-4] for img_file in image_files]  # remove file extension

        # Shuffle image names if shuffle option is enabled
        if self.shuffle:
            random.shuffle(self.image_names)


    def next(self):
        """
        Generates and returns the next batch of images and labels.
        Handles cases where the dataset size isn't a multiple of batch_size.
        """

        # Calculate the starting index of the current batch
        current_image_no = self.current_batch * self.batch_size

        # Determine the end index for the current batch
        last_img_num = current_image_no + self.batch_size
        if last_img_num > len(self.image_names):
            # End of dataset: reshuffle if enabled and update epoch count
            if self.shuffle:
                random.shuffle(self.image_names)
            last_img_num = len(self.image_names)
            self.current_epoch_no += 1
            self.current_batch = 0

        # Select image names for the current batch
        image_batch_names = self.image_names[current_image_no:last_img_num]

        # If batch size isn't met, add images from the beginning to complete the batch
        if len(image_batch_names) < self.batch_size and last_img_num == len(self.image_names):
            num_of_less_images = self.batch_size - len(image_batch_names)
            image_batch_names += self.image_names[:num_of_less_images]

        # Load and process images and labels for the batch
        images = []
        labels = []
        for image_name in image_batch_names:
            image = np.load(self.file_path + image_name + ".npy")
            image = self.augment(image)
            image = cv2.resize(image, tuple(self.image_size[:-1]))
            label = self.labels[image_name]
            images.append(image)
            labels.append(label)

        # Convert lists to numpy arrays for batch processing
        images = np.array(images)
        labels = np.array(labels)

        # Increment batch counter for the next call
        self.current_batch += 1

        return images, labels

    def augment(self, img):

        # Rotate the image if rotation is enabled
        if self.rotation:
            # Select a random rotation angle from 90, 180, or 270 degrees
            angle = random.choice([90, 180, 270])
            # Rotate image by specified angle; angle // 90 gives the number of 90-degree rotations
            img = np.rot90(img, angle // 90, (0, 1))


        # Mirror the image if mirroring is enabled
        if self.mirroring:
            # Flip image along a randomly chosen axis (0: vertical, 1: horizontal)
            img = np.flip(img, random.choice(self.mirror_axis))

        return img

    def current_epoch(self):
        # Return the current epoch number
        return self.current_epoch_no

    def class_name(self, x):
        # Map label index to class name
        return self.class_dict[x]

    def show(self):
        """
        Display a batch of images and their labels for verification.
        This method retrieves the next batch and visualizes it.
        """
        images, labels = self.next()
        
        # Calculate grid size for displaying images
        batch_sqrt = int(np.ceil(np.sqrt(self.batch_size)))  # Ensure batch_sqrt is an integer
        counter = 0
        for image, label in zip(images, labels):
            counter += 1
            plt.subplot(batch_sqrt, batch_sqrt, counter)
            plt.title(self.class_name(label))
            plt.axis('off')
            plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    # Initialize and display an image batch using the generator
    gen = ImageGenerator("exercise_data/", "Labels.json", 60, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)

    gen.show()