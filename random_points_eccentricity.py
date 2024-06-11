import numpy as np
import random
from skimage import measure

# Load the segmented tumor array
tumor_array = np.load(
    "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_111.npy")
test_mask_argmax = np.argmax(tumor_array, axis=3)


# Assuming 'matrix' is your 128x128x128 numpy matrix

# Find contours of the shape in the matrix
contours = measure.find_contours(test_mask_argmax, 0.5)

# Randomly select 8 coordinates on the contours
selected_coords = []
for contour in contours:
    num_points = contour.shape[0]
    if num_points >= 8:
        selected_indices = random.sample(range(num_points), 8)
        for index in selected_indices:
            selected_coords.append(contour[index])

# Convert selected coordinates to integer values
selected_coords = np.round(selected_coords).astype(int)

# Print the selected coordinates
for coord in selected_coords:
    print(coord)

