import numpy as np

# Load the segmented tumor array
tumor_array = np.load("G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_111.npy")
tumor_array = tumor_array[:, :, :, 1]

# Given pixel sizes in millimeters
pixel_size_x = 1.5  # replace with the actual pixel size in the x-dimension
pixel_size_y = 1.5  # replace with the actual pixel size in the y-dimension
pixel_size_z = 4   # replace with the actual pixel size in the z-dimension

# Get the dimensions of the tumor array
num_voxels_x, num_voxels_y, num_voxels_z = 128, 128, 128

# Calculate voxel volume
voxel_volume_mm3 = pixel_size_x * pixel_size_y * pixel_size_z

# Count the number of pixels in the segmented tumor region
number_of_pixels = np.sum(tumor_array > 0)  # Assuming the tumor region is labeled with values greater than 0

# Calculate the tumor volume in mm³
tumor_volume_mm3 = number_of_pixels * voxel_volume_mm3

# Convert tumor volume to cm³
tumor_volume_cm3 = tumor_volume_mm3 / 1000.0

print(f"Voxel Volume: {voxel_volume_mm3:.3f} mm³")
print(f"Tumor Volume: {tumor_volume_mm3:.3f} mm³")
print(f"Tumor Volume: {tumor_volume_cm3:.3f} cm³")
