from ultralytics import YOLO
import os

def main():
    print("==================================================")
    print("  TRAINING CUSTOM YOLOv8 FOR CARLA PERCEPTION")
    print("==================================================")
    
    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Path to our custom dataset YAML
    data_yaml = "/home/conqueror/AI_STORAGE/Town05_Experiment/carla-object-detection-dataset/data.yaml"
    
    if not os.path.exists(data_yaml):
        print(f"Error: Could not find {data_yaml}")
        return
        
    print(f"Dataset configuration found at {data_yaml}")
    print("Starting training on GPU (20 epochs)...")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=20,          # 20 epochs is enough for a small dataset
        imgsz=640,          # Image size
        batch=16,           # Batch size
        device='0',         # Use GPU 0
        name='carla_custom' # Save results to runs/detect/carla_custom
    )
    
    print("==================================================")
    print("  TRAINING COMPLETE")
    print("  Best model saved to runs/detect/carla_custom/weights/best.pt")
    print("==================================================")

if __name__ == "__main__":
    main()
