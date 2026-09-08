import carla
import time
import os
import cv2
import numpy as np

# Create dataset directories
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

# Global variables to store frames
latest_image = None
latest_mask = None
frame_count = 0

def process_rgb(image):
    global latest_image
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_image = i2[:, :, :3].copy()

def process_segmentation(image):
    global latest_mask
    image.convert(carla.ColorConverter.CityScapesPalette)
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_mask = i2[:, :, :3].copy()

def main():
    global frame_count, latest_image, latest_mask
    print("Connecting to CARLA Server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    # Optional: Set weather to clear to get clean images for training
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    blueprint_library = world.get_blueprint_library()
    
    # 1. Spawn Ego Vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Vehicle spawned. Turning on Autopilot!")
    vehicle.set_autopilot(True)
    
    # 2. Attach RGB Camera (for training images)
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    rgb_cam.listen(lambda image: process_rgb(image))
    
    # 3. Attach Semantic Segmentation Camera (for ground-truth labels)
    # Must have the exact same resolution and FOV as RGB!
    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', '800')
    seg_bp.set_attribute('image_size_y', '600')
    seg_bp.set_attribute('fov', '90')
    seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=vehicle)
    seg_cam.listen(lambda image: process_segmentation(image))
    
    print("Cameras attached. Starting Data Collection...")
    
    try:
        # Give cameras time to start up
        time.sleep(2)
        
        while True:
            if latest_image is not None and latest_mask is not None:
                # Save both frames exactly at the same time
                img_path = f"dataset/images/frame_{frame_count:06d}.png"
                mask_path = f"dataset/masks/frame_{frame_count:06d}.png"
                
                cv2.imwrite(img_path, latest_image)
                cv2.imwrite(mask_path, latest_mask)
                
                # Show live feed to the user
                cv2.imshow("Data Collection - RGB", latest_image)
                cv2.imshow("Data Collection - Semantic Mask", latest_mask)
                cv2.waitKey(1)
                
                print(f"Saved Frame {frame_count:06d}")
                frame_count += 1
                
                # Clear frames until next callback
                latest_image = None
                latest_mask = None
                
            time.sleep(0.5) # Collect data every 0.5 seconds to avoid duplicates
            
    except KeyboardInterrupt:
        print("Data Collection Stopped by User.")
    finally:
        print("Destroying actors...")
        rgb_cam.destroy()
        seg_cam.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
