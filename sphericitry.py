import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops, marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Load your npy matrix
tumor_array = np.load(
    "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_111.npy")
matrix = np.argmax(tumor_array, axis=3)
# Extract non-zero values to form a binary mask
binary_mask = matrix > 0

# Find the connected components
labeled_array, num_features = label(binary_mask)

# Calculate volume and surface area for each component
properties = regionprops(labeled_array)

volumes = [prop.area for prop in properties]


# Utility function to calculate mesh surface area
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


# Calculate surface areas using marching cubes
surface_areas = []
for i in range(1, num_features + 1):
    component = (labeled_array == i)
    verts, faces, _, _ = marching_cubes(component, level=0)
    surface_area = mesh_surface_area(verts, faces)
    surface_areas.append(surface_area)


# Compute sphericity for each connected component
def compute_sphericity(volume, surface_area):
    return (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area


sphericities = [compute_sphericity(vol, sa) for vol, sa in zip(volumes, surface_areas)]

# Print the sphericity values
for i, sphericity in enumerate(sphericities):
    print(f'Component {i + 1}: Sphericity = {sphericity}')
