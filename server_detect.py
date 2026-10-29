import cv2
import requests
import base64

API_KEY = "tRVHHVVhltdOXqileCgJ"
# FIXED: Removed the extra '-4' from the project name
MODEL_ID = "mineguard-detection-crele/4"
image_path = r"C:\Users\231348700\Downloads\new sample\dsfd.jpeg"

frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load {image_path}")
    exit()

print("Sending image directly to MineGuard Cloud API...")

# FIXED: Roboflow requires base64 encoding for x-www-form-urlencoded POST requests
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Send the HTTP POST request to Roboflow (50% confidence threshold to match your dashboard)
url = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}&confidence=50&overlap=50"
response = requests.post(
    url,
    data=image_base64,
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

# Catch API errors instead of silently printing "0 objects found"
if response.status_code != 200:
    print(f"\n--- API Error (Status {response.status_code}) ---")
    print(response.text)
    exit()

result = response.json()
predictions = result.get("predictions", [])

if len(predictions) == 0:
    print("\n--- AI Result ---")
    print("The model analyzed the image but found 0 objects.")
else:
    print(f"\n--- AI Result: {len(predictions)} items found ---")
    for pred in predictions:
        class_name = pred["class"]
        confidence = round(pred["confidence"], 2)
        print(f"Detected: {class_name} | Confidence: {int(confidence * 100)}%")

        # Extract center coordinates, width, and height
        cx, cy = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]

        # Convert to [x1, y1, x2, y2]
        x1 = int(cx - (w / 2))
        y1 = int(cy - (h / 2))
        x2 = int(cx + (w / 2))
        y2 = int(cy + (h / 2))

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("MineGuard PPE Detection - Direct API", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()