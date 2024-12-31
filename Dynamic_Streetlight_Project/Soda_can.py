import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_soda_can(image):
    """
    Detects the soda can in the image using color or shape detection.
    Returns the coordinates of the center of the soda can.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find the soda can (largest circular contour)
    for contour in contours:
        # Approximate the contour to a circle
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:  # Threshold to filter out small objects
            return (int(x), int(y))

    return None  # Return None if no soda can is found

# Create Picamera2 instances for two cameras
camera1 = Picamera2(0)  # First camera
camera2 = Picamera2(1)  # Second camera

# Configure the cameras
camera1_config = camera1.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
camera2_config = camera2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})

camera1.configure(camera1_config)
camera2.configure(camera2_config)

# Start the cameras
camera1.start()
camera2.start()

# Capture one frame for calibration
frameVIS = camera1.capture_array()
frameNIR = camera2.capture_array()

# Detect soda can in both images
center1 = detect_soda_can(frameVIS)
center2 = detect_soda_can(frameNIR)

if center1 and center2:
    # Calculate pixel difference between centers
    pixel_difference = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)

    # Print the results
    print(f"Soda can center in VIS: {center1}")
    print(f"Soda can center in NIR: {center2}")
    print(f"Pixel difference between centers: {pixel_difference:.2f} pixels")
else:
    print("Soda can could not be detected in one or both images.")
