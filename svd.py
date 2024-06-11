import numpy as np
import mahotas
import mahotas.demos
import matplotlib.pyplot as plt


def eccentricity(array):
    # Compute the covariance matrix
    covariance_matrix = np.cov(array, rowvar=False)
    # Compute the eigenvalues of the covariance matrix
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    # Calculate the eccentricity
    eccentricity = np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))
    return eccentricity


def printSlice(i):
    array = test_mask_argmax[:, :, i]
    plt.imshow(array)  # cmap='gray' for grayscale, use other colormaps for different looks
    plt.colorbar()  # Add colorbar to show intensity scale
    plt.title("Picture of slice " + str(i))
    plt.show()


def printAllSlices():
    for i in range(0, 128):
        array = test_mask_argmax[:, :, i]
        plt.imshow(array)  # cmap='gray' for grayscale, use other colormaps for different looks
        plt.colorbar()  # Add colorbar to show intensity scale
        plt.title("Picture of slice " + str(i))
        plt.show()


# Load the segmented tumor array
tumor_array = np.load(
    "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_111.npy")
test_mask_argmax = np.argmax(tumor_array, axis=3)

SIGMAS = []
RATIOS = []
RATIOS_FORMULA = []
MAHOTAS_ECC = []
SLOTS_OCCUPIED =[]
# FORMULA_ECC = []
for i in range(128):
    array = test_mask_argmax[:, :, i]
    MAHOTAS_ECC.append(mahotas.features.eccentricity(array))
    # FORMULA_ECC.append(eccentricity(array))
    u, s, v = np.linalg.svd(array)
    if sum(s) == 0:
        RATIOS.append(0)
        RATIOS_FORMULA.append(0)
    else:
        RATIOS.append(s[0] / sum(s))
        RATIOS_FORMULA.append(np.sqrt(1 - (s[1] / s[0]) ** 2))
    SLOTS_OCCUPIED.append(np.count_nonzero(s))
    SIGMAS.append(s)


combined_array = np.column_stack((RATIOS, RATIOS_FORMULA, MAHOTAS_ECC,SLOTS_OCCUPIED))
# Save the combined array to a CSV file
np.savetxt('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/img_111_fullImageSVD.csv', combined_array,
           delimiter=',')

# print(max(RATIOS))
# print(max(MAHOTAS_ECC))
# print(max(FORMULA_ECC))
