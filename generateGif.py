from PIL import Image
import imageio
import os

def combine_images_to_gif(png_folder, output_gif_path):
    # Create a list to store image frames
    frames = []

    # Get the list of PNG files in the folder
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]

    # Sort the files to maintain order
    png_files.sort()

    # Iterate through each PNG file and add it to frames
    for png_file in png_files:
        png_path = os.path.join(png_folder, png_file)
        img = Image.open(png_path)
        frames.append(img)

    # Save frames as a GIF
    imageio.mimsave(output_gif_path, frames, duration=2)

    print(f"Combined {len(frames)} PNG files into {output_gif_path}")

# Example usage
png_folder_path = 'G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/results1/177/original'
output_gif_path = 'G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/results1/177/org2.gif'
combine_images_to_gif(png_folder_path, output_gif_path)
png_folder_path = 'G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/results1/177/mask'
output_gif_path = 'G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/results1/177/mask2.gif'
combine_images_to_gif(png_folder_path, output_gif_path)
png_folder_path = 'G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/results1/177/pred'
output_gif_path = 'G:/Alban & Megi/BrainSegmentation/dataset/BraTS2020_TrainingData/results1/177/pred2.gif'
combine_images_to_gif(png_folder_path, output_gif_path)