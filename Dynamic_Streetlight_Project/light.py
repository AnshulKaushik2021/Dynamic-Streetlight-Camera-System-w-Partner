import requests
import time


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

if __name__ == "__main__":
    DEVICE_ID = "54:70:60:74:F4:F4:4D:04"  
    MODEL = "H6008" 

    try:
        for i in range(1, 10):
            #brightness = i * 19
            #response = control_device(DEVICE_ID, MODEL, "brightness", brightness)
            #print(f"Set Brightness to {brightness}% Response:", response)


            light_intensity = i * 20
            color_response = control_device(DEVICE_ID, MODEL, "color", {"r": light_intensity, "g": light_intensity, "b": light_intensity})
            print(f"Change Color Response:", color_response)

            
            time.sleep(3) 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

