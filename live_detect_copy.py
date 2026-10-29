import cv2
from ultralytics import YOLO

model = YOLO("best.pt")



image_path = r"C:\Users\231348700\Downloads\new sample\dsfd.jpeg"


frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load {image_path}. Make sure the file is in the same folder!")
    exit()

# FIX 1: Lower the confidence to 0.25 (25%) and turn verbose=True so the AI explains itself
results = model.predict(source=frame, conf=0.5, verbose=True)
result = results[0] 

boxes = result.boxes

# FIX 2: Explicitly print if nothing was found
if len(boxes) == 0:
    print("\n--- AI Result ---")
    print("The model analyzed the image but found 0 objects above the 25% confidence threshold.")
    print("Try a different image that clearly shows a Hard_hat, Vest, Person, or Gloves.")
else:
    print(f"\n--- AI Result: {len(boxes)} items found ---")
    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())      
        class_name = result.names[class_id]      
        confidence = round(boxes.conf[i].item(), 2) 
        print(f"Detected: {class_name} | Confidence: {confidence * 100}%")

annotated_frame = result.plot()
cv2.imshow("PPE Detection - Image Test", annotated_frame)

print("\nPress any key in the image window to close it.")
cv2.waitKey(0) 
cv2.destroyAllWindows()