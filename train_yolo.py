from ultralytics import YOLO

# Load a YOLOv8 small model
model = YOLO("yolov8s.pt")

print("Starting YOLOv8 training on CARLA Autopilot Dataset...")
print("This will take approximately 4-6 hours.")

# Train the model
results = model.train(
    data="/home/conqueror/AI_STORAGE/auto_yolo_dataset/data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=16,
    device=0,
    workers=4,
    project="/home/conqueror/AI_STORAGE/yolo_training_results",
    name="carla_yolov8s"
)

print("Training finished successfully!")
