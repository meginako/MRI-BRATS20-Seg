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
    array = test_mask_argmax[:,:, i]
    plt.imshow(array)  # cmap='gray' for grayscale, use other colormaps for different looks
    plt.colorbar()  # Add colorbar to show intensity scale
    plt.title("Picture of slice " + str(i))
    plt.show()
def printAllSlices():
    for i in range(0,128):
        array = test_mask_argmax[:,:, i]
        plt.imshow(array)  # cmap='gray' for grayscale, use other colormaps for different looks
        plt.colorbar()  # Add colorbar to show intensity scale
        plt.title("Picture of slice " + str(i))
        plt.show()

# Load the segmented tumor array
tumor_array = np.load(
    "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_111.npy")
test_mask_argmax = np.argmax(tumor_array, axis=3)
tumor_array = tumor_array[0, :, :, :]
# class 0 has info on first col
# class 1 has info only when we fix the last two cols [i,:,:]
# class 2 yellow??
# class 3 has info only when we fix the last two cols [i,:,:] but too roundy, yellow

#U, S, Vt = np.linalg.svd(tumor_array[:, :, 55])
SIGMAS = []
RATIOS = []
MAHOTAS_ECC = []
#FORMULA_ECC = []
for i in range(128):
    array = test_mask_argmax[:,:, i]
    MAHOTAS_ECC.append(mahotas.features.eccentricity(array))
    #FORMULA_ECC.append(eccentricity(array))
    u, s, v = np.linalg.svd(array)
    if sum(s) == 0:
        RATIOS.append(0)
    else:
        RATIOS.append(s[0] / sum(s))
    SIGMAS.append(s)
print(max(RATIOS))
print(max(MAHOTAS_ECC))
#print(max(FORMULA_ECC))
