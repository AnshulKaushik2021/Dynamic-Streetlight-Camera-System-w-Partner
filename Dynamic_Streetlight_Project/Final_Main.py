import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
from alignment import calibrate_alignment, apply_alignment
from Shadow_Detect import Shadow_Enhance
import requests
import time
import light
import random

API_KEY = "e084e7cb-7399-4d80-bc39-582e8278cbb9"


BASE_URL = "https://developer-api.govee.com/v1"

def control_device(device_id, model, command, value, retries=3):
    """
    Sends a control command to the Govee device with retry logic for rate-limiting.

    :param device_id: The device ID of the Govee light.
    :param model: The model of the Govee device.
    :param command: The command to execute (e.g., brightness, color).
    :param value: The value associated with the command.
    :param retries: Number of retry attempts for handling rate limits.
    :return: The API response as a dictionary.
    """
    url = f"{BASE_URL}/devices/control"
    headers = {
        "Govee-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "device": device_id,
        "model": model,
        "cmd": {
            "name": command,
            "value": value
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.put(url, headers=headers, json=payload)
            

            if response.status_code == 200:
                return response.json()
            

            if response.status_code == 429:
                print("Rate limit reached. Retrying...")
                retry_after = int(response.headers.get("Retry-After", 5))  
                time.sleep(retry_after * (2 ** attempt))  
                continue
            
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            return {"error": str(e), "status_code": response.status_code if response else "N/A"}
    
    return {"error": "Max retries exceeded", "status_code": 429}
# Create Picamera2 instances for two cameras
camera1 = Picamera2(0)  # First camera
camera2 = Picamera2(1)  # Second camera

# Configure the cameras
camera1_config = camera1.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
camera2_config = camera2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})

camera1.configure(camera1_config)
camera2.configure(camera2_config)

#Algo Parameters:
#Default Parameters: alpha=10, gamma=2.2, beta=0.33, tau=10, eta=3.2, neighbors=10, min_theta=0.25
alpha=10
gamma=2.2
beta=0.33
tau=10
eta=3.2
neighbors=10
min_theta=0.3

# Start the cameras
camera1.start()
camera2.start()

# Capture one frame for calibration
frameVIS = camera1.capture_array()
frameNIR = camera2.capture_array()
cv2.imwrite('frameVIS_align.png', frameVIS)
cv2.imwrite('frameNIR_align.png', frameNIR)
#print(frame1[0:10,0:10])
# Calibrate alignment
H_calibrated = calibrate_alignment(frameNIR, frameVIS)

# Create a blank image for the message
height, width = 400, 900  # Adjust size as needed
message_image = np.zeros((height, width, 3), dtype=np.uint8)

# Add text to the image
message = "Press SPACE once filter is applied"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
color = (255, 255, 255)  # White text
thickness = 2
text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
text_x = (width - text_size[0]) // 2
text_y = (height + text_size[1]) // 2
cv2.putText(message_image, message, (text_x, text_y), font, font_scale, color, thickness)

# Display the message
cv2.imshow("Message", message_image)

# Wait for the spacebar press
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # ASCII code for SPACE
        break
def scale(val, old_min = 0.1, old_max = 0.7, new_min= 0, new_max=1):
    scale_value = (val - old_min)/(old_max-old_min) * (new_max - new_min) +new_min
    return max(new_min, min(scale_value, new_max))
# Close the message window
cv2.destroyAllWindows()

#GREEN BOX PARA
# Define the top-left and bottom-right coordinates of the rectangle
top_left = (200, 225)  # (x, y) of the top-left corner
bottom_right = (350, 275)  # (x, y) of the bottom-right corner

# Draw the rectangle
color = (0, 255, 0)  # Green color in BGR format
thickness = 2  # Thickness of the rectangle border (-1 fills the rectangle)
DEVICE_ID = "54:70:60:74:F4:F4:4D:04"  
MODEL = "H6008"
start_time = time.time()
light_cooldown = 5
# Streaming loop for real-time alignment
light_intensity = 5
color_response = control_device(DEVICE_ID, MODEL, "color", {"r": light_intensity, "g": light_intensity, "b": light_intensity})
try:
    while True:
        # Capture frames from both cameras
        frameVIS = camera1.capture_array()
        frameNIR = camera2.capture_array()
        #print(frame1[0:10,0:10])
        # Align the second camera's frame to the first camera's frame
        aligned_frameNIR, aligned_frameVIS = apply_alignment(frameNIR, frameVIS, H_calibrated)
        aligned_frameNIR = cv2.cvtColor(aligned_frameNIR, cv2.COLOR_RGB2GRAY) # FOR NIR PNG THAT IS RGB
        # Combine and display the frames
        combined, ShadowMap = Shadow_Enhance(aligned_frameVIS, aligned_frameNIR, alpha=alpha, gamma=gamma, beta=beta, tau=tau, eta=eta, neighbors=neighbors, min_theta=min_theta)  # Side-by-side comparison
        ShadowMap_255 = ShadowMap * 255
        ShadowMap_bgr = cv2.cvtColor(ShadowMap_255, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(ShadowMap_bgr, top_left, bottom_right, color, thickness)
        
        # Extract the region of interest (ROI)
        roi = ShadowMap[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        print(np.mean(roi))
        DEVICE_ID = "54:70:60:74:F4:F4:4D:04"  
        MODEL = "H6008" 
        # Check if the majority of pixels are white
        
        
        #cv2.imshow("Shadow Map", ShadowMap_bgr)
        cv2.imshow("Combined Image", combined)
        
        # Break on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the cameras and clean up
    camera1.stop()
    camera2.stop()
    cv2.destroyAllWindows()