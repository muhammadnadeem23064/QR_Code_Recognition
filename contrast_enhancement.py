

import os
import cv2
import numpy as np
from cv2 import dnn_superres



def process_images(input_folder_path, output_folder_path):
    # Initialize DNN Super Resolution object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Set the path to the desired model
    model_path = "H:\\QR Code Project\\models\\EDSR_x2.pb"

    # Read the desired model
    sr.readModel(model_path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 2)

    # Define the path to the output folder
    output_folder = os.path.join(output_folder_path, "output_final")

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over all the subfolders and files in the input folder
    for root, dirs, files in os.walk(input_folder_path):
        for file in files:
            # Check if the file is an image file
            if file.endswith('.png') or file.endswith('.jpg'):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Upscale the image
                result = sr.upsample(image)

                # Check if the brightness is less than 112
                brightness = np.mean(result)
                if brightness < 112:
                    # Process the image
                    new_image = np.zeros(result.shape, result.dtype)
                    alpha = 1.3
                    beta = 27
                    for y in range(result.shape[0]):
                        for x in range(result.shape[1]):
                            for c in range(result.shape[2]):
                                new_image[y,x,c] = np.clip(alpha*result[y,x,c] + beta, 0, 255)

                    # Write the processed image to the output folder
                    output_path = os.path.join(output_folder, file)
                    cv2.imwrite(output_path, new_image)

                    # Print a message indicating where the new image was saved
                    print("Processed image saved to:", output_path)
                else:
                    output_path = os.path.join(output_folder, file)
                    cv2.imwrite(output_path, result)

# Call the function with the input and output folder paths
process_images("C:\\Users\\d.arai1798\\Desktop\\Full\\mm", "C:\\Users\\d.arai1798\\Desktop\\Full")










