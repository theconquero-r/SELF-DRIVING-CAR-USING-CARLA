import carla
import numpy as np
import cv2
import os
import time
import queue
import random

# Configuration
SAVE_DIR = '/home/conqueror/SELF_DRIVING_CAR/dataset/segmentation_2'
IMG_DIR = os.path.join(SAVE_DIR, 'images')
MASK_DIR = os.path.join(SAVE_DIR, 'masks')
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

IMG_WIDTH = 1280
IMG_HEIGHT = 720
WAYPOINT_DISTANCE = 10.0  # Collect an image every 10 meters of road

def process_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    return array[:, :, :3] # Return BGR for OpenCV

def process_seg(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    # In CARLA BGRA, the Red channel (index 2) contains the semantic tag
    seg_tags = array[:, :, 2] 
    
    # Create Binary Mask: Road (7) and RoadLines (6) -> 255, else -> 0
    mask = np.zeros_like(seg_tags)
    mask[(seg_tags == 7) | (seg_tags == 6)] = 255
    return mask

def main():
    print("Starting Fast Segmentation Collector...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    
    available_maps = client.get_available_maps()
    # Filter for Town01 to Town07
    towns = [m for m in available_maps if any(t in m for t in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07'])]
    
    # Ensure towns are sorted (optional)
    towns = sorted(towns)
    print(f"Target Towns: {towns}")
    
    total_images = 0
    
    for town_path in towns:
        town_name = town_path.split('/')[-1]
        print(f"\n======================================")
        print(f"Loading {town_name}...")
        client.load_world(town_path)
        world = client.get_world()
        
        # Setup Sync Mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        
        blueprint_library = world.get_blueprint_library()
        
        # We need a dummy actor to attach cameras to (helps with physics/rendering sync in some CARLA versions)
        dummy_bp = blueprint_library.find('static.prop.streetbarrier')
        dummy_bp.set_attribute('role_name', 'dummy')
        dummy_transform = carla.Transform(carla.Location(x=0, y=0, z=-10)) # spawn underground
        dummy = world.try_spawn_actor(dummy_bp, dummy_transform)
        
        if not dummy:
            # Fallback to empty actor
            dummy_bp = blueprint_library.filter('sensor.camera.rgb')[0]
            dummy = world.spawn_actor(dummy_bp, dummy_transform)
        
        # Setup RGB Camera
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(IMG_WIDTH))
        rgb_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
        rgb_bp.set_attribute('fov', '90')
        rgb_cam = world.spawn_actor(rgb_bp, carla.Transform(carla.Location(z=2.4)), attach_to=dummy)
        
        # Setup Segmentation Camera
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(IMG_WIDTH))
        seg_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
        seg_bp.set_attribute('fov', '90')
        seg_cam = world.spawn_actor(seg_bp, carla.Transform(carla.Location(z=2.4)), attach_to=dummy)
        
        rgb_queue = queue.Queue()
        seg_queue = queue.Queue()
        
        rgb_cam.listen(rgb_queue.put)
        seg_cam.listen(seg_queue.put)
        
        # Generate Waypoints
        waypoints = world.get_map().generate_waypoints(WAYPOINT_DISTANCE)
        print(f"Generated {len(waypoints)} waypoints for {town_name}.")
        
        # Weather presets to cycle through for diversity
        weathers = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.HardRainSunset
        ]
        
        for idx, wp in enumerate(waypoints):
            # Change weather every 500 images
            if idx % 500 == 0:
                world.set_weather(random.choice(weathers))
                
            # Base transform
            transform = wp.transform
            
            # Apply Random Perturbation (Drift Recovery)
            right_vec = transform.get_right_vector()
            
            # Random lateral shift between -1.5m (Left) and +1.5m (Right)
            shift = random.uniform(-1.5, 1.5)
            transform.location.x += right_vec.x * shift
            transform.location.y += right_vec.y * shift
            
            # Random yaw rotation (-15 to +15 degrees)
            transform.rotation.yaw += random.uniform(-15.0, 15.0)
            
            # Keep height constant relative to the road
            transform.location.z += 0.5 
            
            # Teleport the dummy actor (cameras move with it)
            dummy.set_transform(transform)
            
            # Tick world
            world.tick()
            
            try:
                rgb_img = rgb_queue.get(timeout=2.0)
                seg_img = seg_queue.get(timeout=2.0)
                
                rgb_frame = process_rgb(rgb_img)
                mask_frame = process_seg(seg_img)
                
                filename = f"{town_name}_frame_{idx:05d}"
                cv2.imwrite(os.path.join(IMG_DIR, f"{filename}.jpg"), rgb_frame)
                cv2.imwrite(os.path.join(MASK_DIR, f"{filename}.png"), mask_frame)
                
                total_images += 1
                
                if idx % 100 == 0:
                    print(f"[{town_name}] Collected: {idx}/{len(waypoints)} | Total Dataset: {total_images}")
                    
            except queue.Empty:
                print(f"Warning: Queue empty at {town_name} waypoint {idx}")
                continue
                
        # Cleanup for next town
        print(f"Finished {town_name}. Destroying actors...")
        rgb_cam.stop()
        seg_cam.stop()
        rgb_cam.destroy()
        seg_cam.destroy()
        dummy.destroy()
        
    print(f"======================================")
    print(f"DATASET COLLECTION COMPLETE!")
    print(f"Total Images Saved: {total_images} in {SAVE_DIR}")
    
if __name__ == '__main__':
    main()
