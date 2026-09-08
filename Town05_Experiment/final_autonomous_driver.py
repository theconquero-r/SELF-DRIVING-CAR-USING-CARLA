import carla
import cv2
import numpy as np
from ultralytics import YOLO
import queue
import pygame

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
MODEL_PATH = "/home/conqueror/AI_STORAGE/yolo_training_results/carla_yolov8s/weights/best.pt"
IMG_WIDTH = 800
IMG_HEIGHT = 600

# Classes from our data.yaml: 0: Pedestrian, 1: Vehicle, 2: Traffic Light, 3: Traffic Sign
CLASSES = {0: 'Pedestrian', 1: 'Vehicle', 2: 'Traffic Light', 3: 'Traffic Sign'}

print("Loading Custom YOLOv8 Model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ==============================================================================
# 2. CARLA CONNECTION
# ==============================================================================
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Set up synchronous mode for smooth FPS
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05 # 20 FPS
world.apply_settings(settings)

# ==============================================================================
# 3. SPAWN EGO VEHICLE & CAMERA
# ==============================================================================
ego_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
ego_vehicle = world.spawn_actor(ego_bp, spawn_point)

# Turn on Autopilot for basic navigation (lane following)
ego_vehicle.set_autopilot(True)

# Attach RGB Camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(IMG_WIDTH))
camera_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

image_queue = queue.Queue()
camera.listen(image_queue.put)

# Setup Pygame for display (Bypassing OpenCV GUI errors)
pygame.init()
display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
pygame.display.set_caption("Custom YOLO Autonomous AI")

print("\nStarting Autonomous Driving AI... Close the window or press 'q' to stop.")
running = True

# ==============================================================================
# 4. MAIN DRIVING LOOP
# ==============================================================================
try:
    while running:
        world.tick()
        image = image_queue.get()
        
        # Convert CARLA image to OpenCV format
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img_array = np.reshape(img_array, (image.height, image.width, 4))
        frame = img_array[:, :, :3].copy() # BGR format
        
        # RUN CUSTOM YOLO INFERENCE
        results = model.predict(frame, verbose=False, conf=0.5)
        
        warning_text = ""
        override_brakes = False
        
        # Analyze detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = CLASSES.get(cls, "Unknown")
                
                # Draw Bounding Box
                color = (0, 255, 0)
                if cls == 0 or cls == 1: # Pedestrian or Vehicle
                    color = (0, 0, 255) # Red for dynamic objects
                    
                    # LOGIC: If a vehicle or pedestrian is large enough (close) and in the center path
                    box_width = x2 - x1
                    box_height = y2 - y1
                    center_x = (x1 + x2) / 2
                    
                    # Check if object is in the middle 50% of the screen and reasonably large
                    in_path = (IMG_WIDTH * 0.25 < center_x < IMG_WIDTH * 0.75)
                    is_close = (box_width > 80 or box_height > 80)
                    
                    if in_path and is_close:
                        override_brakes = True
                        warning_text = f"EMERGENCY BRAKE: {label} AHEAD!"
                        
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # APPLY CONTROLS
        if override_brakes:
            # Override autopilot and slam the brakes
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False))
            cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Let Autopilot drive normally
            cv2.putText(frame, "AUTOPILOT ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the camera feed using Pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        display.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

finally:
    print("\nShutting down and cleaning up...")
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    
    if camera:
        camera.destroy()
    if ego_vehicle:
        ego_vehicle.destroy()
    pygame.quit()
    print("Done.")
