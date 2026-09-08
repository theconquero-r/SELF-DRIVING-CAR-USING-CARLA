import carla
import time
import random
import numpy as np
import cv2
import queue
import os
import sys

# Directory Setup
DATASET_DIR = "/home/conqueror/AI_STORAGE/auto_yolo_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images/train")
LBL_DIR = os.path.join(DATASET_DIR, "labels/train")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    f.write(f"path: {DATASET_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/train\n\n")
    f.write("nc: 4\n")
    f.write("names:\n")
    f.write("  0: pedestrian\n")
    f.write("  1: vehicle\n")
    f.write("  2: traffic_light\n")
    f.write("  3: traffic_sign\n")

IMG_WIDTH = 640
IMG_HEIGHT = 640
FOV = 110
FRAMES_PER_TOWN = 1500 

def spawn_npc(world, blueprint_library, spawn_points, tm, tm_port):
    vehicles_list = []
    vehicle_bps = blueprint_library.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4 or int(x.get_attribute('number_of_wheels')) == 2]
    
    for _ in range(50):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        bp.set_attribute('role_name', 'autopilot')
        spawn_point = random.choice(spawn_points)
        npc = world.try_spawn_actor(bp, spawn_point)
        if npc is not None:
            npc.set_autopilot(True, tm_port)
            vehicles_list.append(npc)
            
    walkers_list = []
    controllers_list = []
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    
    for _ in range(50):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point = carla.Transform(loc)
            walker_bp = random.choice(walker_bps)
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            if walker:
                walkers_list.append(walker)
                
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    for walker in walkers_list:
        controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
        if controller:
            controllers_list.append(controller)
            
    world.tick()
    for controller in controllers_list:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.4 + random.random())
        
    print(f"Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} pedestrians.")
    return vehicles_list, walkers_list, controllers_list

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    import pygame
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
    pygame.display.set_caption("Live Auto-Labeling View")
    
    # Resuming from Town07 (Final Town!)
    towns_to_visit = ['Town07']
    
    # Dynamically find the correct saved_count so we never overwrite old images!
    existing_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    saved_count = len(existing_images)
    
    print("\n" + "="*50)
    print("  MULTI-TOWN AUTO-LABELING GENERATOR (WITH LIVE VIEW)")
    print(f"  Target: {FRAMES_PER_TOWN} frames per town.")
    print("="*50 + "\n")
    
    for town in towns_to_visit:
        print(f"\n---> Loading {town}...")
        world = client.load_world(town)
        time.sleep(2.0)
        
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.global_percentage_speed_difference(10.0)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        ego = None
        vehicles_list = []
        walkers_list = []
        controllers_list = []
        rgb_cam = None
        sem_cam = None
        
        try:
            ego_bp = blueprint_library.filter("model3")[0]
            ego_bp.set_attribute('role_name', 'ego')
            ego = world.spawn_actor(ego_bp, random.choice(spawn_points))
            ego.set_autopilot(True, 8000)
            tm.ignore_lights_percentage(ego, 0)
            
            vehicles_list, walkers_list, controllers_list = spawn_npc(world, blueprint_library, spawn_points, tm, 8000)
            
            cam_bp = blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            cam_bp.set_attribute('fov', str(FOV))
            
            sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
            sem_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            sem_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            sem_bp.set_attribute('fov', str(FOV))
            
            cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            
            rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
            sem_cam = world.spawn_actor(sem_bp, cam_transform, attach_to=ego)
            
            rgb_queue = queue.Queue()
            sem_queue = queue.Queue()
            
            rgb_cam.listen(rgb_queue.put)
            sem_cam.listen(sem_queue.put)
            
            frames_in_this_town = 0
            frame_count = 0
            
            # Map YOLO class ID to a LIST of CityScapes BGR Colors
            # Format: B, G, R
            TARGET_COLORS_BGR = {
                0: [(60, 20, 220), (0, 0, 255)],   # Pedestrian, Rider
                1: [(142, 0, 0), (70, 0, 0), (100, 60, 0), (230, 0, 0), (32, 11, 119)], # Car, Truck, Bus, Motorcycle, Bicycle
                2: [(30, 170, 250)],  # Traffic Light
                3: [(0, 220, 220)]    # Traffic Sign
            }
            
            # Colors for Pygame drawing (R, G, B)
            COLORS = {
                0: (0, 255, 255), # Cyan
                1: (0, 128, 255), # Blue-ish
                2: (255, 0, 0),   # Red
                3: (255, 0, 255)  # Magenta
            }
            
            CLASS_NAMES = ["Pedestrian", "Vehicle", "Traffic Light", "Traffic Sign"]
            
            while frames_in_this_town < FRAMES_PER_TOWN:
                world.tick()
                
                rgb_data = rgb_queue.get()
                sem_data = sem_queue.get()
                
                if frame_count % 5 == 0:
                    rgb_array = np.frombuffer(rgb_data.raw_data, dtype=np.dtype("uint8"))
                    rgb_array = np.reshape(rgb_array, (rgb_data.height, rgb_data.width, 4))
                    rgb_frame = rgb_array[:, :, :3].copy()
                    
                    # ULTIMATE FIX: Convert to CityScapes palette to get standard, reliable colors
                    # This bypasses all raw channel bugs in CARLA.
                    sem_data.convert(carla.ColorConverter.CityScapesPalette)
                    sem_array = np.frombuffer(sem_data.raw_data, dtype=np.dtype("uint8"))
                    sem_array = np.reshape(sem_array, (sem_data.height, sem_data.width, 4))
                    
                    yolo_annotations = []
                    display_frame = rgb_frame.copy()
                    
                    # Ultimate Foolproof Bounding Box Extraction using CityScapes Colors
                    for cls_id, target_bgr_list in TARGET_COLORS_BGR.items():
                        mask = np.zeros((sem_data.height, sem_data.width), dtype=np.uint8)
                        
                        for target_bgr in target_bgr_list:
                            sub_mask = (sem_array[:, :, 0] == target_bgr[0]) & \
                                       (sem_array[:, :, 1] == target_bgr[1]) & \
                                       (sem_array[:, :, 2] == target_bgr[2])
                            mask |= sub_mask.astype(np.uint8)
                        
                        # Find distinct blobs of this class
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                        
                        # Skip 0 because 0 is the background (not the object)
                        for i in range(1, num_labels):
                            x, y, w, h, area = stats[i]
                            
                            # Ignore tiny noise blobs
                            if area < 100: continue
                            
                            # Ignore ego vehicle hood (huge blob at bottom center)
                            if cls_id == 1 and (y + h) > IMG_HEIGHT * 0.9 and w > IMG_WIDTH * 0.4:
                                continue
                                
                            # Convert to YOLO format
                            cx = (x + w / 2.0) / IMG_WIDTH
                            cy = (y + h / 2.0) / IMG_HEIGHT
                            nw = w / float(IMG_WIDTH)
                            nh = h / float(IMG_HEIGHT)
                            
                            yolo_annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                            
                            # Draw live view
                            color = COLORS[cls_id]
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(display_frame, CLASS_NAMES[cls_id], (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # LIVE VIEW
                    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                    display.blit(surface, (0, 0))
                    pygame.display.flip()
                    pygame.event.pump()
                    
                    if len(yolo_annotations) > 0:
                        frame_name = f"frame_{saved_count:06d}"
                        cv2.imwrite(os.path.join(IMG_DIR, f"{frame_name}.jpg"), rgb_frame)
                        with open(os.path.join(LBL_DIR, f"{frame_name}.txt"), "w") as f:
                            f.write("\n".join(yolo_annotations))
                        
                        saved_count += 1
                        frames_in_this_town += 1
                        
                        sys.stdout.write(f"\r[{town}] Saved {frames_in_this_town}/{FRAMES_PER_TOWN} (Total: {saved_count}) | Found: {len(yolo_annotations)} objects")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\r[{town}] Processing frame {frame_count}... Searching for objects...")
                        sys.stdout.flush()
                        
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopped by user.")
            pygame.quit()
            sys.exit(0)
        finally:
            print(f"\nCleaning up {town}...")
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
            try:
                if rgb_cam: rgb_cam.destroy()
            except: pass
            try:
                if sem_cam: sem_cam.destroy()
            except: pass
            try:
                if ego: ego.destroy()
            except: pass
            
            for v in vehicles_list:
                try: v.destroy()
                except: pass
            for c in controllers_list:
                try:
                    c.stop()
                    c.destroy()
                except: pass
            for w in walkers_list:
                try: w.destroy()
                except: pass
            
    pygame.quit()
    print(f"\nALL DONE! Total {saved_count} perfectly labeled images collected from 5 towns!")

if __name__ == '__main__':
    main()
