import carla
import time
import cv2
import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()
display = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Antigravity Lag-Free Test Drive")
clock = pygame.time.Clock()

latest_image = None
image_updated = False

def process_rgb(image):
    global latest_image, image_updated
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_image = i2[:, :, :3].copy()
    image_updated = True

def main():
    print("Connecting to CARLA Server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0) # Increased timeout to 30 seconds for map loading
    
    # We won't force map loading here anymore.
    # We will let the user launch the server directly into the desired map to save VRAM!
    world = client.get_world()
    
    # Set good lighting
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)
    
    blueprint_library = world.get_blueprint_library()
    
    # Spawn Ego Vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    import random
    random.shuffle(spawn_points)
    spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0, y=0, z=2))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Ego Vehicle spawned!")
    
    # Spawn Traffic (5 Vehicles instead of 20 to save MASSIVE CPU Physics load)
    print("Spawning 5 AI traffic vehicles for testing...")
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    
    auto_bps = [x for x in blueprint_library.filter('vehicle.*') if int(x.get_attribute('number_of_wheels')) == 4]
    spawned_traffic = []
    
    for i in range(1, min(6, len(spawn_points))):
        bp = random.choice(auto_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
            
        npc = world.try_spawn_actor(bp, spawn_points[i])
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            spawned_traffic.append(npc)
    print(f"Spawned {len(spawned_traffic)} traffic vehicles!")
    
    # We will use Asynchronous mode (like yesterday) because Sync mode locks FPS to 20!
    print("Using Asynchronous Mode for maximum Pygame FPS...")
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.0333 # 30 FPS physics step (fixes visual stuttering)
    world.apply_settings(settings)
    
    traffic_manager.set_synchronous_mode(False)
    
    # Attach RGB Camera (Optimized resolution for max performance)
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=-6.5, z=3.0), carla.Rotation(pitch=-15.0))
    rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    
    # Listen async
    rgb_cam.listen(process_rgb)
    
    control = carla.VehicleControl()
    
    # Pygame Setup (Now just a tiny window for keyboard input)
    pygame.init()
    display = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("CLICK HERE TO DRIVE")
    clock = pygame.time.Clock()
    
    running = True
    try:
        while running:
            clock.tick(60) # Run pygame smoothly at 60 FPS
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            keys = pygame.key.get_pressed()
            
            # Throttle & Brake
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                control.throttle = min(control.throttle + 1.0, 1.0)
                control.brake = 0.0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                control.throttle = 0.0
                control.brake = min(control.brake + 1.0, 1.0)
            else:
                control.throttle = max(control.throttle - 0.1, 0.0)
                control.brake = 0.0
                
            # Steering
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                control.steer = max(control.steer - 0.1, -1.0)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                control.steer = min(control.steer + 0.1, 1.0)
            else:
                if control.steer > 0:
                    control.steer = max(control.steer - 0.1, 0.0)
                elif control.steer < 0:
                    control.steer = min(control.steer + 0.1, 0.0)
                    
            # Reverse
            control.reverse = keys[pygame.K_q]
            
            vehicle.apply_control(control)
            
            # Fill pygame window to indicate focus
            display.fill((50, 50, 200))
            font = pygame.font.SysFont(None, 24)
            text = font.render(f"Speed: {int(vehicle.get_velocity().length() * 3.6)} km/h", True, (255, 255, 255))
            display.blit(text, (20, 20))
            text2 = font.render(f"KEEP THIS WINDOW FOCUSED", True, (255, 255, 0))
            display.blit(text2, (20, 50))
            pygame.display.flip()
            
            # Ultra-Optimized Hardware Rendering using OpenCV (Bypasses Pygame bottleneck completely)
            global image_updated
            if image_updated and latest_image is not None:
                cv2.imshow("Antigravity Dashcam (Hardware Accelerated)", latest_image)
                cv2.waitKey(1)
                image_updated = False # Reset flag until next camera frame
                
    finally:
        print("Cleaning up actors and resetting world...")
        cv2.destroyAllWindows()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)
        
        rgb_cam.destroy()
        vehicle.destroy()
        
        for npc in spawned_traffic:
            if npc is not None and npc.is_alive:
                npc.destroy()
        print("All traffic cleaned up!")
        
        pygame.quit()

if __name__ == '__main__':
    main()
