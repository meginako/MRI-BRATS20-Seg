import csv
import os

import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops, marching_cubes


def add_row_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)


def mesh_surface_area(verts, faces):
    area = 0
    for face in faces:
        tri_verts = verts[face]
        side_a = np.linalg.norm(tri_verts[0] - tri_verts[1])
        side_b = np.linalg.norm(tri_verts[1] - tri_verts[2])
        side_c = np.linalg.norm(tri_verts[2] - tri_verts[0])
        s = (side_a + side_b + side_c) / 2
        area += np.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
    return area


def calculate(file_name):
    # Load your npy matrix
    tumor_array = np.load(
        "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/" + file_name)
    matrix = np.argmax(tumor_array, axis=3)
    # Assuming the voxel size is 1x1x1 in arbitrary units, e.g., millimeters
    voxel_volume = 1  # mm^3
    voxel_area = 1  # mm^2
    # Extract non-zero values to form a binary mask
    binary_mask = matrix > 0
    # Find the connected components
    labeled_array, num_features = label(binary_mask)
    # Calculate properties for each component
    properties = regionprops(labeled_array)
    # Function to calculate mesh surface area
    # Loop through each component and calculate parameters
    for prop in properties:
        volume = prop.area * voxel_volume  # Volume in mm^3
        coords = prop.coords
        # Create a binary mask for the component
        component_mask = (labeled_array == prop.label)
        # Calculate surface area using marching cubes
        verts, faces, _, _ = marching_cubes(component_mask, level=0.5)
        surface_area = mesh_surface_area(verts, faces) * voxel_area  # Surface area in mm^2
        # Calculate bounding box
        bbox = prop.bbox
        bbox_volume = ((bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (
                bbox[5] - bbox[2])) * voxel_volume  # Bounding box volume in mm^3
        # Calculate sphericity
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
        # Print the results with units
        print(f'Component {prop.label}:')
        print(f'  Volume: {volume:.2f} mm^3')
        print(f'  Surface Area: {surface_area:.2f} mm^2')
        print(f'  Sphericity: {sphericity:.4f}')
        print(f'  Bounding Box Volume: {bbox_volume:.2f} mm^3')
        print(f'  Extent: {volume / bbox_volume:.4f}')
        new_data = [volume, surface_area, sphericity, bbox_volume, volume / bbox_volume, file_name]
        add_row_to_csv('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/3DShapeFeatures.csv', new_data)


new_data = ['volume', 'surface_area', 'sphericity', 'bbox_volume', 'extent', 'file_name']
add_row_to_csv('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/3DShapeFeatures.csv', new_data)

directory_path = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks"

# Get all the file names in the directory
file_names = os.listdir(directory_path)

for file_name in file_names:
    calculate(file_name)
