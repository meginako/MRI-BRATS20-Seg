import csv

import numpy as np
import random
from skimage import measure
import matplotlib.pyplot as plt
import os

def eccentricityRandomPoints(npy_array, nr_rand):
    '''
    # Step 1: Visualize the npy array
    plt.imshow(npy_array, cmap='gray')
    plt.title('Original Array')
    plt.show()
    '''
    # Step 2: Find the contours
    contours = measure.find_contours(npy_array, level=0.5)

    # Find the largest contour
    # Check if there is only one contour
    if len(contours) == 1:
        largest_contour = contours[0]
    elif len(contours) == 0:
        return 0
    else:
        # Find the largest contour
        largest_contour = max(contours, key=len)

    # check if the contuour is really small or not
    if len(largest_contour) < nr_rand:
        return 0
    '''
    # Step 3: Plot the contours and highlight the largest contour
    plt.imshow(npy_array, cmap='gray')
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=2, color='red')
    plt.title('Contours with Largest Contour Highlighted')
    plt.show()
    '''
    # Step 4: Generate 8 random points on the largest contour
    random_indices = random.sample(range(len(largest_contour)), nr_rand)
    random_points = largest_contour[random_indices]

    # Plot the random points on the largest contour
    '''
    plt.imshow(npy_array, cmap='gray')
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=2, color='red')
    plt.scatter(random_points[:, 1], random_points[:, 0], color='blue')
    plt.title(str(nr_rand) + 'Random Points on Largest Contour')
    plt.show()
    '''
    # Step 5: Compute the SVD
    # Center the points by subtracting the mean
    points_centered = random_points - np.mean(random_points, axis=0)
    U, S, Vt = np.linalg.svd(points_centered)

    # Step 6: Calculate the eccentricity
    # Eccentricity = sqrt(1 - (minor_axis / major_axis)^2)
    eccentricity = np.sqrt(1 - (S[1] / S[0]) ** 2)
    #print("Eccentricity:", eccentricity)

    return eccentricity

def add_row_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)

directory_path = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks"

# Get all the file names in the directory
file_names = os.listdir(directory_path)

for file_name in file_names:
    # Load the segmented tumor array
    tumor_array = np.load(
        "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/"+file_name)
    test_mask_argmax = np.argmax(tumor_array, axis=3)

    EccentrictiyRandom16Points = []
    EccentrictiyRandom32Points = []
    EccentrictiyRandom64Points = []

    for i in range(128):
        EccentrictiyRandom16Points.append(eccentricityRandomPoints(test_mask_argmax[:, :, i], 16))
        EccentrictiyRandom32Points.append(eccentricityRandomPoints(test_mask_argmax[:, :, i], 32))
        EccentrictiyRandom64Points.append(eccentricityRandomPoints(test_mask_argmax[:, :, i], 64))



    combined_array = np.column_stack((EccentrictiyRandom16Points, EccentrictiyRandom32Points, EccentrictiyRandom64Points))
    # Save the combined array to a CSV file
    np.savetxt('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/'+os.path.splitext(file_name)[0]+'_randomPoints.csv', combined_array,
               delimiter=',')


    if EccentrictiyRandom16Points:
        overall_eccentricity16 = np.mean(EccentrictiyRandom16Points)
        overall_eccentricity32 = np.mean(EccentrictiyRandom32Points)
        overall_eccentricity64 = np.mean(EccentrictiyRandom64Points)
        print("Overall Eccentricity for the 3D shape with 16 points:", overall_eccentricity16)
        print("Overall Eccentricity for the 3D shape with 32 points:", overall_eccentricity32)
        print("Overall Eccentricity for the 3D shape with 64 points:", overall_eccentricity64)
        new_data = [overall_eccentricity16, overall_eccentricity32, overall_eccentricity64, file_name]
        add_row_to_csv('G:/Alban & Megi/BrainSegmentation/dataset/predictedNPYResults/ECCENTRICITY3D.csv', new_data)
    else:
        print("No valid contours found in the 3D array.")

    # empty the arrays
    EccentrictiyRandom16Points = []
    EccentrictiyRandom32Points = []
    EccentrictiyRandom64Points = []

