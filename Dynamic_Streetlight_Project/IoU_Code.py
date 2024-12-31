import os
import numpy as np
import cv2
from PIL import Image
from Shadow_Detect_Final import Shadow_Detect, Shadow_Enhance
from sklearn.metrics import jaccard_score

def trim_images(predicted_image, ground_truth_image):
    
    # Determine the minimum dimensions
    min_rows = min(predicted_image.shape[0], ground_truth_image.shape[0])
    min_cols = min(predicted_image.shape[1], ground_truth_image.shape[1])
    
    # Trim both images to the minimum dimensions
    predicted_trimmed = predicted_image[:min_rows, :min_cols]
    ground_truth_trimmed = ground_truth_image[:min_rows, :min_cols]
    
    return predicted_trimmed, ground_truth_trimmed

# Define input folders and output folder
rgb_folder = "./Shadow/TestImages"
nir_folder = "./Shadow/TestImages"
sha_folder = "./Shadow/TestImages"
output_folder = "./Shadow/TestImages_Final"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through all RGB images in the RGB folder
for file_name in os.listdir(rgb_folder):
    if file_name.startswith("VIS"):  # Match the RGB file naming pattern
        # Remove the "VIS" prefix to generate the base name
        base_name = file_name.replace("VIS", "", 1)
        
        # Ensure base_name retains only the core file name without extensions
        if base_name.endswith(".png"):  # Assumes files have the '.png' extension
            base_name = base_name[:-4]
        
        # Generate the corresponding NIR and SHA file names
        nir_file_name = f"NIR{base_name}.png"
        sha_file_name = f"SHA{base_name}.png"
        
        # Define full file paths
        rgb_path = os.path.join(rgb_folder, file_name)
        nir_path = os.path.join(nir_folder, nir_file_name)
        sha_path = os.path.join(sha_folder, sha_file_name)
        output_path = os.path.join(output_folder, f"SHA{base_name}_mine.jpg")
        
        # Ensure both RGB and NIR files exist
        if os.path.exists(rgb_path) and os.path.exists(nir_path):
            # Load images
            imageVIS = Image.open(rgb_path)
            imageNIR = Image.open(nir_path)
            imageSHA = Image.open(sha_path)

            
            # Convert images to NumPy arrays
            VIS = np.array(imageVIS)
            NIR = np.array(imageNIR)
            SHA = np.array(imageSHA)

            NIR = cv2.cvtColor(NIR, cv2.COLOR_RGB2GRAY)
            # Apply the enhancement algorithm
            U_bin, _ = Shadow_Detect(VIS, NIR) 
            U_bin = U_bin * 255

            SHA = cv2.cvtColor(SHA, cv2.COLOR_RGB2GRAY)

            # Ensure both images have the same dimensions
            predicted_image, ground_truth_image = trim_images(U_bin, SHA)

            predicted_image = (predicted_image > 127).astype(np.uint8)

            ground_truth_image = (ground_truth_image > 127).astype(np.uint8)

            y_pred = predicted_image.flatten()
            y_true = ground_truth_image.flatten()
            iou = jaccard_score(y_true, y_pred)
            print(iou)
        
            # Convert RGB to BGR for OpenCV compatibility
            U_bin = cv2.cvtColor(U_bin, cv2.COLOR_GRAY2BGR)
            
            # Save the enhanced image
            cv2.imwrite(output_path, U_bin)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Missing files for: {base_name}")



