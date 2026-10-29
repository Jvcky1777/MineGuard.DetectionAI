from ultralytics import YOLO

# Load your local weights file
model = YOLO("best.pt")

# Extract the exact base model used during training
base_model = model.ckpt.get("train_args", {}).get("model", "Unknown")
print(f"This model was trained using: {base_model}")