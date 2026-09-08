import carla
import time
import os
import cv2
import numpy as np
import pygame

# Initialize Pygame for manual control
pygame.init()
display = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Manual Data Collector (WASD to Drive)")
clock = pygame.time.Clock()

# Create dataset directories
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

# Global variables
latest_image = None
latest_mask = None
frame_count = 0
last_save_time = 0

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
    global frame_count, latest_image, latest_mask, last_save_time
    print("Connecting to CARLA Server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    # Set weather to clear to get clean images for training
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    blueprint_library = world.get_blueprint_library()
    
    # 1. Spawn Ego Vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Vehicle spawned. Get ready to drive manually!")
    
    # 2. Attach RGB Camera
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    rgb_cam.listen(lambda image: process_rgb(image))
    
    # 3. Attach Semantic Segmentation Camera
    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', '800')
    seg_bp.set_attribute('image_size_y', '600')
    seg_bp.set_attribute('fov', '90')
    seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=vehicle)
    seg_cam.listen(lambda image: process_segmentation(image))
    
    # Control state
    control = carla.VehicleControl()
    
    print("Cameras attached. Data Collection will start in 2 seconds...")
    time.sleep(2)
    
    running = True
    print("DRIVE USING W-A-S-D or ARROW KEYS. Press ESC to stop and save.")
    
    try:
        while running:
            clock.tick(60) # 60 FPS Pygame loop
            
            # 1. Handle Keyboard Inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            keys = pygame.key.get_pressed()
            
            # Throttle (W or UP)
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                control.throttle = min(control.throttle + 0.1, 1.0)
            else:
                control.throttle = max(control.throttle - 0.1, 0.0)
                
            # Brake (S or DOWN)
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                control.brake = min(control.brake + 0.2, 1.0)
            else:
                control.brake = 0.0
                
            # Steer (A/D or LEFT/RIGHT)
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                control.steer = max(control.steer - 0.05, -1.0)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                control.steer = min(control.steer + 0.05, 1.0)
            else:
                # Return steering to center smoothly
                if control.steer > 0:
                    control.steer = max(control.steer - 0.05, 0.0)
                elif control.steer < 0:
                    control.steer = min(control.steer + 0.05, 0.0)
                    
            # Reverse gear
            control.reverse = keys[pygame.K_q]
            
            # Apply control to CARLA vehicle
            vehicle.apply_control(control)
            
            # 2. Render RGB image to Pygame Window for driving
            if latest_image is not None:
                # Pygame uses RGB, OpenCV uses BGR. We need to swap channels.
                img_surface = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)
                # Rotate and flip for pygame coordinate system
                img_surface = np.rot90(img_surface)
                img_surface = np.flipud(img_surface)
                
                surface = pygame.surfarray.make_surface(img_surface)
                display.blit(surface, (0, 0))
                
                # Draw text overlay
                font = pygame.font.SysFont(None, 36)
                text = font.render(f"Frames Collected: {frame_count} / 2000+", True, (0, 255, 0))
                display.blit(text, (20, 20))
                
                pygame.display.flip()
            
            # 3. Save Data (Every 0.2 seconds -> 5 FPS dataset collection)
            current_time = time.time()
            if current_time - last_save_time > 0.2:
                if latest_image is not None and latest_mask is not None:
                    # Append instead of overwrite by taking the max existing frame number
                    img_path = f"dataset/images/frame_{frame_count:06d}.png"
                    mask_path = f"dataset/masks/frame_{frame_count:06d}.png"
                    
                    cv2.imwrite(img_path, latest_image)
                    cv2.imwrite(mask_path, latest_mask)
                    
                    frame_count += 1
                    last_save_time = current_time
                    
                    # Optional: Small print to console
                    if frame_count % 50 == 0:
                        print(f"[{frame_count}/2000] Frames collected...")
                        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Stopping Data Collection and cleaning up...")
        rgb_cam.destroy()
        seg_cam.destroy()
        vehicle.destroy()
        pygame.quit()
        print(f"Successfully collected a total of {frame_count} custom frames!")

if __name__ == '__main__':
    main()
