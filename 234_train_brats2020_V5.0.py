


# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu

Code to train batches of cropped BraTS 2020 images using 3D U-net.

Please get the data ready and define custom data gnerator using the other
files in this directory.

Images are expected to be 128x128x128x3 npy data (3 corresponds to the 3 channels for
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 128x128x128x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""

import os
import numpy as np
from custom_datagen import imageLoader
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random

####################################################
train_img_dir = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/train/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0, num_images - 1)
test_img = np.load(train_img_dir + img_list[img_num])
test_mask = np.load(train_mask_dir + msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

#############################################################
# Optional step of finding the distribution of each class and calculating appropriate weights
# Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

import pandas as pd

columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)
train_mask_list = sorted(glob.glob('G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/train/masks/*.npy'))
for img in range(len(train_mask_list)):
    print(img)
    temp_image = np.load(train_mask_list[img])
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)
    zipped = zip(columns, counts)
    conts_dict = dict(zipped)

    # # df = df.append(conts_dict, ignore_index=True)
    # temp_image_tensor = tf.convert_to_tensor(np.load(train_mask_list[img]))
    #
    # # Use torch.argmax along the appropriate axis (axis=3)
    # temp_image_argmax = tf.argmax(temp_image_tensor, dim=3)
    #
    # # Convert to NumPy array if needed
    # temp_image_argmax_np = temp_image_argmax.numpy()
    #
    # # Calculate unique values and their counts
    # val, counts = tf.unique(temp_image_argmax, return_counts=True)

    # Convert to NumPy arrays for further processing if needed
    # val_np, counts_np = val.numpy(), counts.numpy()

    # Create a dictionary from the counts
    # conts_dict = dict(zip(columns, counts))

    # Convert conts_dict values to torch tensors
    # conts_dict = {key: tf.tensor(value) for key, value in conts_dict.items()}

    df = df.append(conts_dict, ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['1'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4
# Class weights claculation: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels / (n_classes * label_0)), 2)  # round to 2 decimals
wt1 = round((total_labels / (n_classes * label_1)), 2)
wt2 = round((total_labels / (n_classes * label_2)), 2)
wt3 = round((total_labels / (n_classes * label_3)), 2)

# Weights are: 0.26, 22.53, 22.53, 26.21
# wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
# These weihts can be used for Dice loss

##############################################################
# Define the image generators for training and validation

train_img_dir = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/train/masks/"

val_img_dir = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
##################################

########################################################################
batch_size = 8

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

###########################################################################
# Define loss, metrics and optimizer to be used for training

wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
import segmentation_models_3D as sm
import tensorflow as ts

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.95)]

LR = 0.0001
# optim = ts.keras.optimizers.Adam(LR)
optim = ts.keras.optimizers.Adam(LR)

#######################################################################
# Fit the model

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

from simple_3d_unet import simple_unet_model
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callbacks_list = [early_stopping]

model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=3,
                          num_classes=4)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)

import time

# Record start time
start_time = time.time()

# Your model training code here
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks_list)

# Record end time
end_time = time.time()

# Calculate total training time
total_time = end_time - start_time

# Print the total training time
print(f"Total training time: {total_time} seconds")

import pandas as pd
# Convert the history to a DataFrame
history_df = pd.DataFrame(history.history)

# Define the CSV file path
csv_file_path = 'saved_models/v11/training_history.csv'

# Save the DataFrame to a CSV file
history_df.to_csv(csv_file_path, index=False)

print(f"Training history saved to {csv_file_path}")

model.save('saved_models/v11/brats_3d_v5.hdf5')
##################################################################


# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################
from keras.models import load_model

# Load model for prediction or continue training

# For continuing training....
# The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
# This is because the model does not save loss function and metrics. So to compile and
# continue training we need to provide these as custom_objects.
my_model = load_model('G:/Alban & Megi/BrainSegmentation/brats_3d.hdf5')

# So let us add the loss as custom object... but the following throws another error...
# Unknown metric function: iou_score
my_model = load_model('G:/Alban & Megi/BrainSegmentation/brats_3d.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# Now, let us add the iou_score function we used during our initial training
my_model = load_model('brats_3d.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5)})

# Now all set to continue the training process.
history2 = my_model.fit(train_img_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=1,
                        verbose=1,
                        validation_data=val_img_datagen,
                        validation_steps=val_steps_per_epoch,
                        )
#################################################


import cv2

import numpy as np
from keras.models import load_model
from keras.metrics import MeanIoU
# Load the model
my_model = load_model('saved_models/v1/brats_3d_v5.hdf5', compile=False)

# Function to apply Laplacian filter to an image
def apply_laplacian(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    return laplacian

# Load test image and mask
img_num = 15
test_img = np.load("G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/images/image_" + str(img_num) + ".npy")
test_mask = np.load("G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/input_data_128/val/masks/mask_" + str(img_num) + ".npy")

# Apply Laplacian filter to the test image
test_img_filtered = apply_laplacian(test_img)

# Expand dimensions for model input
test_img_input = np.expand_dims(test_img_filtered, axis=0)

# Make predictions
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

# Calculate IoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_prediction_argmax, np.argmax(test_mask, axis=3))
print("Mean IoU =", IOU_keras.result().numpy())

# Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(np.argmax(test_mask, axis=3)[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.show()