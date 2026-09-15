import cv2
from ultralytics import YOLO

model = YOLO("best.pt")#loading the trained model

cap = cv2.VideoCapture(0)#capturing the video from webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Camera initialized. Press 'q' to quit the program.")

while True:
    success, frame = cap.read()#reading the frames from webcam
    if not success:
        print("Error: Could not read frame.")
        break
    results = model.predict(source=frame, conf=0.5, verbose=False)#predicting the objects in the frame

    result = results[0] 

    # Extract the raw data out of the bounding boxes
    boxes = result.boxes

    # Loop through every object detected in this specific frame
    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())      # What is it?
        class_name = result.names[class_id]      # Translate ID to string name
        confidence = round(boxes.conf[i].item(), 2) # How sure is it?
        
        # Print the data to your terminal
        print(f"Detected: {class_name} | Confidence: {confidence * 100}%")

    # Plotting the visual results on the frame
    annotated_frame = result.plot()
    
    cv2.imshow("Live PPE Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()#releasing the webcam
cv2.destroyAllWindows()#destroying all the windows