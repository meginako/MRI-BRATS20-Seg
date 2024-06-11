import numpy as np
from scipy.ndimage import label
from skimage.measure import perimeter
def perimeterAndArea(array):
    # Label the shapes in the array
    labeled_array, num_features = label(array > 0)  # label shapes with values > 0
    perimeters = []
    areas = []
    # Calculate the perimeter of each shape
    for i in range(1, num_features + 1):
        shape_mask = labeled_array == i
        shape_perimeter = perimeter(shape_mask, neighbourhood=8)
        shape_area = np.sum(shape_mask)
        perimeters.append(shape_perimeter)
        areas.append(shape_area)
    return sum(perimeters), sum(areas)


# Load the segmented tumor array
tumor_array = np.load(
    "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_111.npy")
test_mask_argmax = np.argmax(tumor_array, axis=3)
Perimeters = []
Area = []
EccentricityPA = []
for i in range(128):
    p, a = perimeterAndArea(test_mask_argmax[:, :, i])
    Perimeters.append(p)
    Area.append(a)
    if p == 0:
        EccentricityPA.append(0)
    else:
        EccentricityPA.append(a/p)



combined_array = np.column_stack((Perimeters, Area, EccentricityPA))
# Save the combined array to a CSV file
np.savetxt('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/img_111.csv', combined_array, delimiter=',')

