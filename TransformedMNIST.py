import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from skimage.util import random_noise

class TransformedMNIST:
    def __init__(self):
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None
        self.training_validation_images = None
        self.training_validation_labels = None

        print("Loading MNIST dataset...")
        (self.training_images, self.training_labels), (self.test_images, self.test_labels) = mnist.load_data()
        self.training_images, self.training_validation_images, self.training_labels, self.training_validation_labels = train_test_split(self.training_images, self.training_labels, test_size=0.2, random_state=0)
        print("MNIST dataset loaded...")
        
        self.modification_proportion = 0.25
        self.sample_size = len(self.training_validation_images)
        self.transformed_images_count =  int(self.sample_size * self.modification_proportion)

        self.rotationDrifitImagesSet = copy.deepcopy(self.training_validation_images)
        self.scaleDrifitImagesSet = copy.deepcopy(self.training_validation_images)
        self.noisedDrifitImagesSet = copy.deepcopy(self.training_validation_images)
        
        self.rotatedImagesIndices = self.rotate_images(30, 270)
        self.scaledImagesIndices = self.scale_images(1.8)
        self.noisyImagesIndices = self.noise_images('gaussian')

    
    def select_images_to_modify(self, images_set):
        indices_to_modify = np.random.choice(len(images_set), self.transformed_images_count, replace=False)
        return indices_to_modify
    
    def rotate_images(self, min_rotation=30, max_rotation=270):
        print(f"Rotating images -> min_rotation: {min_rotation}, max_rotation: {max_rotation} proportion: {self.modification_proportion:.2f}% affected images: {self.transformed_images_count} out of {self.sample_size}")
        indices_to_rotate = self.select_images_to_modify(self.rotationDrifitImagesSet)
        for i in indices_to_rotate:
            angle = np.random.uniform(-min_rotation, max_rotation)
            self.rotationDrifitImagesSet[i] = self.rotate_image(self.rotationDrifitImagesSet[i], angle)
        return indices_to_rotate
    
    def scale_images(self, scaling_factor=1.8):
        print(f"Scaling images -> scaling_factor: {scaling_factor} proportion: {self.modification_proportion:.2f}% affected images: {self.transformed_images_count} out of {self.sample_size}")
        indices_to_scale = self.select_images_to_modify(self.scaleDrifitImagesSet)
        for i in indices_to_scale:
            self.scaleDrifitImagesSet[i] = self.scale_image(self.scaleDrifitImagesSet[i], scaling_factor)
        return indices_to_scale

    def noise_images(self, noise_type='gaussian'):
        print(f"Adding noise to images -> noise_type: {noise_type} proportion: {self.modification_proportion:.2f}% affected images: {self.transformed_images_count} out of {self.sample_size}")
        indices_to_noise = self.select_images_to_modify(self.noisedDrifitImagesSet)
        for i in indices_to_noise:
            self.noisedDrifitImagesSet[i] = self.add_noise(self.noisedDrifitImagesSet[i], noise_type)
        return indices_to_noise
    
    def change_backgrounds(self):
        print(f"Changing backgrounds proportion: {self.modification_proportion:.2f}% affected images: {self.transformed_images_count} out of {self.sample_size}")
        indices_to_change_background = self.select_images_to_modify(self.training_validation_images)
        for i in indices_to_change_background:
            self.training_validation_images[i] = self.change_background_of_image(self.training_validation_images[i])
        return indices_to_change_background

    @staticmethod
    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    @staticmethod
    def scale_image(image, factor):
        height, width = image.shape
        new_height, new_width = int(height * factor), int(width * factor)
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Centered crop or pad the image to 28x28
        if factor > 1:
            # Calculate the cropping coordinates
            start_x = max(0, (new_width - 28) // 2)
            start_y = max(0, (new_height - 28) // 2)
            end_x = start_x + 28
            end_y = start_y + 28
            cropped = scaled[start_y:end_y, start_x:end_x]
            return cropped
        else:
            # Pad the scaled image if it's smaller than 28x28
            pad_width_x = (28 - new_width) // 2
            pad_width_y = (28 - new_height) // 2
            return np.pad(scaled, ((pad_width_y, 28 - new_height - pad_width_y), 
                (pad_width_x, 28 - new_width - pad_width_x)), 'constant', constant_values=(0, 0))

    @staticmethod
    def add_noise(image, noise_type):
        return random_noise(image, mode=noise_type)
    
    @staticmethod
    def change_background_of_image(image, noise_intensity=25):
        # Creating a mask for the background (pixels with zero intensity)
        background_mask = (image == 0)
        # Adding uniform noise to the background
        noise = np.random.randint(0, noise_intensity, image.shape)
        image_with_noise = image.copy()
        image_with_noise[background_mask] += noise[background_mask]
        # Ensuring pixel values remain in the valid range
        image_with_noise = np.clip(image_with_noise, 0, 255)
        return image_with_noise


    
    def plot_change_matrix(self):
        num_samples = 5  # Number of samples to display
        # Randomly select indices for the samples to display for each change type
        selected_noisy_indices = np.random.choice(self.noisyImagesIndices, num_samples, replace=False)
        selected_scaled_indices = np.random.choice(self.scaledImagesIndices, num_samples, replace=False)
        selected_rotated_indices = np.random.choice(self.rotatedImagesIndices, num_samples, replace=False)

        # 6 columns: 3 for original images and 3 for modified images (noise, scale, rotation)
        fig, axs = plt.subplots(num_samples, 6, figsize=(18, num_samples * 3))  

        for i in range(num_samples):
            # Column 1: Original image for noise
            noisy_idx = selected_noisy_indices[i]
            axs[i, 0].imshow(self.training_validation_images[noisy_idx], cmap='gray')
            axs[i, 0].set_title(f'Original (Idx: {noisy_idx})')
            axs[i, 0].axis('off')

            # Column 2: Noised image
            axs[i, 1].imshow(self.noisedDrifitImagesSet[noisy_idx], cmap='gray')
            axs[i, 1].set_title(f'Noised')
            axs[i, 1].axis('off')

            # Column 3: Original image for scale
            scaled_idx = selected_scaled_indices[i]
            axs[i, 2].imshow(self.training_validation_images[scaled_idx], cmap='gray')
            axs[i, 2].set_title(f'Original (Idx: {scaled_idx})')
            axs[i, 2].axis('off')

            # Column 4: Scaled image
            axs[i, 3].imshow(self.scaleDrifitImagesSet[scaled_idx], cmap='gray')
            axs[i, 3].set_title(f'Scaled')
            axs[i, 3].axis('off')

            # Column 5: Original image for rotation
            rotated_idx = selected_rotated_indices[i]
            axs[i, 4].imshow(self.training_validation_images[rotated_idx], cmap='gray')
            axs[i, 4].set_title(f'Original (Idx: {rotated_idx})')
            axs[i, 4].axis('off')

            # Column 6: Rotated image
            axs[i, 5].imshow(self.rotationDrifitImagesSet[rotated_idx], cmap='gray')
            axs[i, 5].set_title(f'Rotated')
            axs[i, 5].axis('off')

        plt.tight_layout()
        plt.show()
        
        # Print the indices of the changed images for each type
        np.set_printoptions(threshold=np.inf)  # Set threshold to infinity to print the whole array

        print("Rotated images indices:", self.rotatedImagesIndices)
        print("Scaled images indices:", self.scaledImagesIndices)
        print("Noisy images indices:", self.noisyImagesIndices)

        np.set_printoptions(threshold=1000)  # Reset to default threshold
        
