import cv2
import numpy as np

def calibrate_alignment(img1, img2):
    import cv2
    import numpy as np

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Match descriptors using BFMatcher with KNN
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Ensure enough matches
    if len(matches) < 4:
        raise ValueError("Not enough good matches to compute homography. Found only {}.".format(len(matches)))
    
    # Use good matches for homography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H



def apply_alignment(imgChg, imgRef, H):
    # Warp the image using the stored homography
    height, width, _ = imgRef.shape
    aligned_imgChg = cv2.warpPerspective(imgChg, H, (width, height))
    
    # Convert both images to grayscale for intersection
    aligned_imgChg_gray = cv2.cvtColor(aligned_imgChg, cv2.COLOR_BGR2GRAY)
    imgRef_gray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    
    # Create binary masks for non-black areas
    mask_aligned = aligned_imgChg_gray > 0
    mask_ref = imgRef_gray > 0
    
    # Ensure masks have the same shape and type
    mask_aligned = mask_aligned.astype(np.uint8)
    mask_ref = mask_ref.astype(np.uint8)
    
    # Compute the intersection mask
    intersection_mask = cv2.bitwise_and(mask_aligned, mask_ref)
    
    # Find the bounding box of the intersection region
    x, y, w, h = cv2.boundingRect(intersection_mask)
    
    # Crop both images to the intersection region
    cropped_imgChg = aligned_imgChg[y:y+h, x:x+w]
    cropped_imgRef = imgRef[y:y+h, x:x+w]
    
    return cropped_imgChg, cropped_imgRef


# Example usage
# Load initial pair of images for calibration
imageVIS1 = cv2.imread('norm_pre_align.jpg')
imageNIR1 = cv2.imread('noir_pre_align.jpg')

# Calibrate alignment
H_calibrated = calibrate_alignment(imageNIR1, imageVIS1)

# Load a new pair of images
imageVIS2 = cv2.imread('norm_pre_align.jpg')
imageNIR2 = cv2.imread('noir_pre_align.jpg')

# Align the new images using the stored homography
aligned_VIS2, aligned_NIR2 = apply_alignment(imageNIR2, imageVIS2, H_calibrated)
# Save or process the aligned images
cv2.imwrite('aligned_VIS2.jpg', aligned_VIS2)
cv2.imwrite('aligned_NIR2.jpg', aligned_NIR2)
