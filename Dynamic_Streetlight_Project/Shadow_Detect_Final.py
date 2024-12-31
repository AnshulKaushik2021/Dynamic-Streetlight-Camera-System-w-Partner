import numpy as np
#import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

def sigmoid_transform(x, alpha, beta, gamma):
    x = x / 255.0  # Normalize pixel values to [0, 1]
    x_transformed = 1 / (1 + np.exp(-alpha * (1 - x**(1/gamma) - beta)))
    return (x_transformed * 255).astype(np.uint8)  # Rescale to [0, 255]


def find_first_valley_variable(hist, neighbors=2, min_theta=0.25):
    """
    Finds the first valley in the histogram after a specified fraction of the histogram range.

    Parameters:
        hist (list or array): The histogram values.
        neighbors (int): The number of neighboring bins to consider for valley detection.
        min_theta (float): Minimum fraction of the histogram range to consider (default is 0.1).

    Returns:
        int: The index of the first valley, or None if no valley is found.
    """
    min_index = int(len(hist) * min_theta)  # Minimum index to start searching for valleys
    
    for i in range(max(neighbors, min_index), len(hist) - neighbors):
        is_valley = all(hist[i] < hist[i - j] and hist[i] < hist[i + j] for j in range(1, neighbors + 1))
        if is_valley:
            return i
    return None


def find_threshold_with_increment(image_array, initial_bins, neighbors, min_theta=0.25, max_iterations=10):
    # Mask non-zero values to ignore black pixels
    mask = image_array > 0
    image_array = image_array[mask]
    
    bins = initial_bins
    for iteration in range(max_iterations):
        # Compute histogram with the current number of bins
        hist, bin_edges = np.histogram(image_array, bins=bins)
        
        # Find the first valley
        valley_index = find_first_valley_variable(hist, neighbors, min_theta=min_theta)
        if valley_index is not None:
            return bin_edges[valley_index], bins  # Return threshold and bins used
        
        # If no valley found, increase the number of bins
        bins = int(bins * 1.2)  # Increment bins by 20%
    
    return None, bins  # Return None if no valley found after max_iterations



def Shadow_Detect (VIS, NIR, alpha=10, gamma=2.2, beta=0.33, tau=10, eta=3.2, neighbors=10, min_theta=0.25):
    # Normalize each channel to [0, 1] (sensor response)
    VIS_NORM = VIS.astype(np.float32) / 255.0
    NIR_NORM = NIR.astype(np.float32) / 255.0
    NIR_NORM[NIR_NORM == 0] = 1e-6

    # Calculate the brightness image L
    L = np.mean(VIS_NORM, axis=2)  # Average of R, G, and B channels

    # Scale L back to [0, 255] for visualization
    VIS_L = (L * 255).astype(np.uint8)
    #Candidate Map
    # Apply the sigmoid transformation
    D_VIS = sigmoid_transform(VIS_L, alpha, beta, gamma)
    D_NIR = sigmoid_transform(NIR, alpha, beta, gamma)
    
    D = (D_VIS.astype(np.float32) / 255.0) * (D_NIR.astype(np.float32) / 255.0)
    D = (D * 255).astype(np.uint8)  # Scale back to [0, 255]

    #Ratio Image
    # Compute the ratio t^k_ij for each channel (R, G, B)
    t_r = VIS_NORM[:, :, 2] / NIR_NORM  # Red channel ratio
    t_g = VIS_NORM[:, :, 1] / NIR_NORM  # Green channel ratio
    t_b = VIS_NORM[:, :, 0] / NIR_NORM  # Blue channel ratio

    # Compute t_ij using the min(max(...)) from paper
    T = 1 / tau * np.minimum(np.maximum.reduce([t_r, t_g, t_b]), tau)

    #Shadow Map
    U = (1-D/255)*(1-T)

    m, n = np.shape(U)
    N_bins = int(eta * np.ceil(np.log2(m*n) + 1))

    # Find threshold with incrementing bins
    theta, _ = find_threshold_with_increment(U, N_bins, neighbors, min_theta=min_theta)

    # Perform thresholding to create a binary shadow mask
    U_bin = (U <= theta).astype(np.uint8)

    return U_bin, U

def Shadow_Enhance(VIS, NIR, alpha=10, gamma=2.2, beta=0.33, tau=10, eta=3.2, neighbors=10, min_theta=0.25):
    U_bin, U = Shadow_Detect(VIS, NIR, alpha=alpha, gamma=gamma, beta=beta, tau=tau, eta=eta, neighbors=neighbors, min_theta=min_theta)
    
    # Create the valid region mask
    #valid_mask = np.logical_and(np.any(VIS > 0, axis=2), NIR > 0)

    valid_mask = NIR > 0

    
    VIS_HSV = cv2.cvtColor(VIS, cv2.COLOR_RGB2HSV)  # Convert to HSV
    VIS_V = VIS_HSV[:, :, 2]
    
    VIS_V_norm = VIS_V / 255.0
    NIR_norm = NIR / 255.0
    
    # Apply the mask to exclude invalid regions
    #VIS_V_norm[~valid_mask] = 0
    NIR_norm[~valid_mask] = 0
    U[~valid_mask] = 0
    U_bin[~valid_mask] = 0
    
    # Enhance only valid regions
    BLENDED_V = U * VIS_V_norm + (1 - U) * NIR_norm
    BLENDED_V[~valid_mask] = VIS_V_norm[~valid_mask]  # Keep invalid regions as black
    
    BLENDED_V_255 = (BLENDED_V * 255).astype(np.uint8)

    blended_hsv = VIS_HSV.copy()
    blended_hsv[:, :, 2] = BLENDED_V_255

    blended_rgb = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2RGB)

    return blended_rgb, U_bin


#Example Usage, After alignment
# Load images
# NIR = cv2.imread('aligned_NIR2.jpg')
# VIS = cv2.imread('aligned_VIS2.jpg')
# NIR = cv2.imread('noir_post_align.jpg')
# VIS = cv2.imread('norm_post_align.jpg')

# NIR = cv2.cvtColor(NIR, cv2.COLOR_RGB2GRAY)

# NEW_VIS, Shadow_Map = Shadow_Enhance(VIS,NIR)
# Shadow_Map = Shadow_Map * 255
# cv2.imwrite('shadow_enhance_post_align.jpg', NEW_VIS)
# cv2.imwrite('shadow_map_post_align.jpg', Shadow_Map)
