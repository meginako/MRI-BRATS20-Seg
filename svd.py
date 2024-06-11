import csv

import numpy as np
import mahotas
import mahotas.demos
import os

'''
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
'''


def add_row_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)


def calculate(file_name):
    # Load the segmented tumor array
    tumor_array = np.load(
        "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/" + file_name)
    test_mask_argmax = np.argmax(tumor_array, axis=3)

    SIGMAS = []
    RATIOS = []
    RATIOS_FORMULA = []
    MAHOTAS_ECC = []
    SLOTS_OCCUPIED = []
    for i in range(128):
        array = test_mask_argmax[:, :, i]
        MAHOTAS_ECC.append(mahotas.features.eccentricity(array))
        u, s, v = np.linalg.svd(array)
        if sum(s) == 0:
            RATIOS.append(0)
            RATIOS_FORMULA.append(0)
        else:
            RATIOS.append(s[0] / sum(s))
            RATIOS_FORMULA.append(np.sqrt(1 - (s[1] / s[0]) ** 2))
        SLOTS_OCCUPIED.append(np.count_nonzero(s))
        SIGMAS.append(s)

    combined_array = np.column_stack((RATIOS, RATIOS_FORMULA, MAHOTAS_ECC, SLOTS_OCCUPIED))
    # Save the combined array to a CSV file
    np.savetxt('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/SvdSlices' + os.path.splitext(file_name)[
        0] + '.csv', combined_array,
               delimiter=',')

    new_data = [np.mean(RATIOS), np.mean(RATIOS_FORMULA), np.mean(MAHOTAS_ECC), file_name]
    add_row_to_csv('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/MeanEccentricityAllSliceSVD.csv',
                   new_data)

directory_path = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks"

# Get all the file names in the directory
file_names = os.listdir(directory_path)

for file_name in file_names:
    calculate(file_name)