import carla
import pygame
import cv2
import os
import math
import random
import csv
import json
import numpy as np
import queue
import sys

# Ensure CARLA PythonAPI is in path for GlobalRoutePlanner
carla_api_path = "/home/conqueror/AI_STORAGE/carla_packed_linux/CARLA_0.9.16/PythonAPI/carla"
if carla_api_path not in sys.path:
    sys.path.append(carla_api_path)
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

# --- Configuration ---
SAVE_DIR = '/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Omni'
IMG_DIR = os.path.join(SAVE_DIR, 'images')
MASK_DIR = os.path.join(SAVE_DIR, 'masks')
LIDAR_DIR = os.path.join(SAVE_DIR, 'lidar')
LABEL_DIR = os.path.join(SAVE_DIR, 'labels') # For YOLO Ground Truth

for d in [IMG_DIR, MASK_DIR, LIDAR_DIR, LABEL_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_FRAMES = 15000 
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FOV = 90.0

# Command Mapping
CMD_MAP = {
    RoadOption.LEFT: 0, RoadOption.RIGHT: 1, RoadOption.STRAIGHT: 2,
    RoadOption.LANEFOLLOW: 3, RoadOption.CHANGELANELEFT: 3,
    RoadOption.CHANGELANERIGHT: 3, RoadOption.VOID: 3
}

# --- Camera Math for 3D-to-2D Projection ---
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)
    
    # UE4 to Standard Camera: (X forward, Y right, Z up) -> (X right, Y down, Z forward)
    # So: point_camera = [Y, -Z, X]
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    
    # now project 3D->2D using the camera matrix
    if point_camera[2] > 0: # If in front of the camera
        point_img = np.dot(K, point_camera)
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
        return int(point_img[0]), int(point_img[1]), point_camera[2]
    return -1, -1, -1

def get_2d_bounding_box(actor, K, w2c, img_w, img_h):
    if not hasattr(actor, 'bounding_box'): return None
    bb = actor.bounding_box
    verts = bb.get_world_vertices(actor.get_transform())
    
    points = []
    for v in verts:
        x, y, d = get_image_point(v, K, w2c)
        if d > 0:
            points.append((x, y))
            
    if len(points) < 8: # Requires all 8 vertices of the 3D box to be in front of the camera
        return None
        
    pts = np.array(points)
    x_min = max(0, np.min(pts[:, 0]))
    x_max = min(img_w - 1, np.max(pts[:, 0]))
    y_min = max(0, np.min(pts[:, 1]))
    y_max = min(img_h - 1, np.max(pts[:, 1]))
    
    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)
    return None

def extract_segmentation_boxes(seg_tags, class_id):
    mask = (seg_tags == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5: # Filter out 1-pixel noise
            boxes.append((x, y, x + w, y + h))
    return boxes

def process_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    return array[:, :, :3].copy()

def process_seg(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    seg_tags = array[:, :, 2]
    mask = np.zeros_like(seg_tags)
    mask[(seg_tags == 7) | (seg_tags == 6)] = 255
    return mask.copy()

def process_lidar(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points

def detect_traffic_light_color(rgb_image, x1, y1, x2, y2):
    cropped = rgb_image[y1:y2, x1:x2]
    if cropped.size == 0: return "UNKNOWN"
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    _, glowing_mask = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)
    h, w = glowing_mask.shape
    if h < w: return "UNKNOWN"
    third = h // 3
    top_px = cv2.countNonZero(glowing_mask[0:third, :])
    mid_px = cv2.countNonZero(glowing_mask[third:2*third, :])
    bot_px = cv2.countNonZero(glowing_mask[2*third:h, :])
    max_px = max(top_px, mid_px, bot_px)
    if max_px < 5: return "UNKNOWN"
    if max_px == top_px: return "RED"
    elif max_px == mid_px: return "YELLOW"
    else: return "GREEN"

def main():
    print("Initializing CARLA API Ground Truth Omni-Model Data Collector (Town05)...")
    pygame.init()
    display = pygame.display.set_mode((1280, 360))
    pygame.display.set_caption("Omni Collector (API Ground Truth)")
    font = pygame.font.SysFont('Arial', 20, bold=True)
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    
    town_name = "Town05"
    print(f"Loading {town_name}...")
    client.load_world(town_name)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(10.0)
    
    blueprint_library = world.get_blueprint_library()
    
    print("Cleaning up zombie vehicles...")
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
    vehicle.set_autopilot(True, 8000)
    
    npc_vehicles = []
    npc_blueprints = blueprint_library.filter('vehicle.*')
    ego_loc = vehicle.get_location()
    close_spawn_points = [p for p in spawn_points if 20.0 < p.location.distance(ego_loc) < 250.0]
    random.shuffle(close_spawn_points)
    
    for i in range(min(50, len(close_spawn_points))):
        npc_bp = random.choice(npc_blueprints)
        npc = world.try_spawn_actor(npc_bp, close_spawn_points[i])
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npc_vehicles.append(npc)
    print(f"Spawned {len(npc_vehicles)} NPC vehicles!")
    
    # Sensors (NVIDIA 3-Camera Setup)
    # Center Camera
    cam_transform_center = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4), carla.Rotation(pitch=0.0))
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IMG_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
    rgb_bp.set_attribute('fov', str(FOV))
    rgb_cam_center = world.spawn_actor(rgb_bp, cam_transform_center, attach_to=vehicle)
    
    # Left Camera (Y = -0.5m)
    cam_transform_left = carla.Transform(carla.Location(x=1.5, y=-0.5, z=2.4), carla.Rotation(pitch=0.0))
    rgb_cam_left = world.spawn_actor(rgb_bp, cam_transform_left, attach_to=vehicle)
    
    # Right Camera (Y = 0.5m)
    cam_transform_right = carla.Transform(carla.Location(x=1.5, y=0.5, z=2.4), carla.Rotation(pitch=0.0))
    rgb_cam_right = world.spawn_actor(rgb_bp, cam_transform_right, attach_to=vehicle)
    
    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', str(IMG_WIDTH))
    seg_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
    seg_bp.set_attribute('fov', str(FOV))
    seg_cam = world.spawn_actor(seg_bp, cam_transform_center, attach_to=vehicle)
    
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_cam = world.spawn_actor(lidar_bp, cam_transform_center, attach_to=vehicle)
    
    rgb_queue_center = queue.Queue()
    rgb_queue_left = queue.Queue()
    rgb_queue_right = queue.Queue()
    seg_queue = queue.Queue()
    lidar_queue = queue.Queue()
    
    rgb_cam_center.listen(rgb_queue_center.put)
    rgb_cam_left.listen(rgb_queue_left.put)
    rgb_cam_right.listen(rgb_queue_right.put)
    seg_cam.listen(seg_queue.put)
    lidar_cam.listen(lidar_queue.put)
    
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    route = []
    
    csv_file = open(os.path.join(SAVE_DIR, 'telemetry.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame_ID', 'Speed', 'Steer', 'Throttle', 'Brake', 'Command', 'GPS_X', 'GPS_Y', 'GPS_Z', 'IMU_Pitch', 'IMU_Yaw', 'IMU_Roll', 'Ego_TL_State', 'YOLO_JSON'])
    
    K = build_projection_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)
    
    frame_count = 0
    running = True
    
    print("Starting Collection Loop...")
    try:
        while running and frame_count < TARGET_FRAMES:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            ego_loc = vehicle.get_location()
            
            # --- Dynamic Routing ---
            if len(route) < 10:
                dest = random.choice(spawn_points).location
                route = grp.trace_route(ego_loc, dest)
                try:
                    path_locations = [wp[0].transform.location for wp in route]
                    tm.set_path(vehicle, path_locations)
                except Exception:
                    pass 
                    
            min_dist = float('inf')
            closest_idx = 0
            for i in range(min(20, len(route))):
                dist = ego_loc.distance(route[i][0].transform.location)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            route = route[closest_idx:]
            current_command = CMD_MAP.get(route[0][1], 3) if len(route) > 0 else 3
            
            # --- Steering Noise (Drift Recovery) ---
            if frame_count % 30 == 0 and frame_count > 10:
                control = vehicle.get_control()
                noise = random.uniform(-0.4, 0.4)
                control.steer = np.clip(control.steer + noise, -1.0, 1.0)
                vehicle.apply_control(control)
                
            world.tick()
            
            try:
                rgb_img_center = rgb_queue_center.get(timeout=2.0)
                rgb_img_left = rgb_queue_left.get(timeout=2.0)
                rgb_img_right = rgb_queue_right.get(timeout=2.0)
                seg_img = seg_queue.get(timeout=2.0)
                lidar_data = lidar_queue.get(timeout=2.0)
                
                rgb_frame_center = process_rgb(rgb_img_center)
                rgb_frame_left = process_rgb(rgb_img_left)
                rgb_frame_right = process_rgb(rgb_img_right)
                mask_frame = process_seg(seg_img)
                lidar_points = process_lidar(lidar_data)
                
                # Create a copy for the UI so we don't burn boxes into the training data!
                ui_frame = rgb_frame_center.copy()
                
                # 1. --- CARLA API 3D Intelligence (Vehicles & Pedestrians) ---
                camera_transform = rgb_cam_center.get_transform()
                w2c = np.array(camera_transform.get_inverse_matrix())
                
                all_actors = world.get_actors()
                vehicles = all_actors.filter('vehicle.*')
                pedestrians = all_actors.filter('walker.pedestrian.*')
                
                detected_objects = [] # For JSON
                yolo_labels = []      # For YOLO txt
                
                def process_actor_list(actor_list, class_id, class_name):
                    for actor in actor_list:
                        if actor.id == vehicle.id: continue # Skip ego vehicle
                        if actor.get_location().distance(ego_loc) > 50.0: continue
                        
                        bbox = get_2d_bounding_box(actor, K, w2c, IMG_WIDTH, IMG_HEIGHT)
                        if bbox is not None:
                            x_min, y_min, x_max, y_max = bbox
                            
                            cx = (x_min + (x_max - x_min) / 2) / IMG_WIDTH
                            cy = (y_min + (y_max - y_min) / 2) / IMG_HEIGHT
                            w = (x_max - x_min) / IMG_WIDTH
                            h = (y_max - y_min) / IMG_HEIGHT
                            
                            obj_dict = {"class": class_name, "box": [round(cx,4), round(cy,4), round(w,4), round(h,4)]}
                            detected_objects.append(obj_dict)
                            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            
                            cv2.rectangle(ui_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

                process_actor_list(vehicles, 0, 'vehicle')
                process_actor_list(pedestrians, 1, 'pedestrian')
                
                # 2. --- SEMANTIC SEGMENTATION Intelligence (Traffic Lights & Signs) ---
                # This guarantees PIXEL-PERFECT bounding boxes directly from the graphics engine
                raw_seg = np.frombuffer(seg_img.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
                seg_tags = raw_seg[:, :, 2]
                
                # In this CARLA version's palette, Road=1, TrafficLight=7, TrafficSign=8
                tl_boxes = extract_segmentation_boxes(seg_tags, 7) # Traffic Light = 7
                sign_boxes = extract_segmentation_boxes(seg_tags, 8) # Traffic Sign = 8
                
                def process_seg_boxes(boxes, class_id, class_name, box_color):
                    for bbox in boxes:
                        x_min, y_min, x_max, y_max = bbox
                        cx = (x_min + (x_max - x_min) / 2) / IMG_WIDTH
                        cy = (y_min + (y_max - y_min) / 2) / IMG_HEIGHT
                        w = (x_max - x_min) / IMG_WIDTH
                        h = (y_max - y_min) / IMG_HEIGHT
                        
                        obj_dict = {"class": class_name, "box": [round(cx,4), round(cy,4), round(w,4), round(h,4)]}
                        
                        # Add specific state if traffic light
                        draw_color = box_color
                        if class_name == 'traffic_light':
                            detected_color = detect_traffic_light_color(rgb_frame_center, int(x_min), int(y_min), int(x_max), int(y_max))
                            obj_dict["color"] = detected_color
                            if detected_color == "RED": draw_color = (0, 0, 255)
                            elif detected_color == "GREEN": draw_color = (0, 255, 0)
                            elif detected_color == "YELLOW": draw_color = (0, 255, 255)
                            else: draw_color = (255, 255, 255) # White for UNKNOWN (e.g. back of the light)
                            
                        detected_objects.append(obj_dict)
                        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        cv2.rectangle(ui_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), draw_color, 2)
                        
                process_seg_boxes(tl_boxes, 2, 'traffic_light', (0, 255, 255))
                process_seg_boxes(sign_boxes, 3, 'traffic_sign', (0, 0, 255))
                
                yolo_json_str = json.dumps(detected_objects)
                
                # Fetch Telemetry
                control = vehicle.get_control()
                vel = vehicle.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
                
                transform = vehicle.get_transform()
                gps_x = transform.location.x
                gps_y = transform.location.y
                gps_z = transform.location.z
                imu_pitch = transform.rotation.pitch
                imu_yaw = transform.rotation.yaw
                imu_roll = transform.rotation.roll
                
                # Get Ego Traffic Light State from API Cheat for CSV
                ego_tl_state = "NONE"
                if vehicle.is_at_traffic_light():
                    state = vehicle.get_traffic_light_state()
                    if state == carla.TrafficLightState.Red: ego_tl_state = "RED"
                    elif state == carla.TrafficLightState.Yellow: ego_tl_state = "YELLOW"
                    elif state == carla.TrafficLightState.Green: ego_tl_state = "GREEN"
                
                # --- Save Data (NVIDIA 3-Camera Format) ---
                filename = f"frame_{frame_count:05d}"
                cv2.imwrite(os.path.join(IMG_DIR, f"{filename}_center.jpg"), rgb_frame_center)
                cv2.imwrite(os.path.join(IMG_DIR, f"{filename}_left.jpg"), rgb_frame_left)
                cv2.imwrite(os.path.join(IMG_DIR, f"{filename}_right.jpg"), rgb_frame_right)
                cv2.imwrite(os.path.join(MASK_DIR, f"{filename}.png"), mask_frame)
                np.save(os.path.join(LIDAR_DIR, f"{filename}.npy"), lidar_points)
                
                # Save YOLO Ground Truth Label
                with open(os.path.join(LABEL_DIR, f"{filename}.txt"), 'w') as f:
                    f.write("\\n".join(yolo_labels))
                
                csv_writer.writerow([filename, speed, control.steer, control.throttle, control.brake, current_command, gps_x, gps_y, gps_z, imu_pitch, imu_yaw, imu_roll, ego_tl_state, yolo_json_str])
                
                # --- Render Pygame UI (Center Camera) ---
                disp_rgb = cv2.resize(ui_frame, (640, 360))
                disp_mask = cv2.resize(mask_frame, (640, 360))
                disp_mask_color = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2RGB)
                disp_rgb_pygame = cv2.cvtColor(disp_rgb, cv2.COLOR_BGR2RGB)
                
                combined = np.hstack((disp_rgb_pygame, disp_mask_color))
                surface = pygame.surfarray.make_surface(combined.swapaxes(0, 1))
                display.blit(surface, (0, 0))
                
                text_str = f"Town 5 GT API: {frame_count}/{TARGET_FRAMES} | Objects: {len(detected_objects)} | Cmd: {current_command}"
                text_surface = font.render(text_str, True, (0, 255, 0))
                pygame.draw.rect(display, (0, 0, 0), (8, 8, text_surface.get_width() + 4, text_surface.get_height() + 4))
                display.blit(text_surface, (10, 10))
                
                pygame.display.flip()
                frame_count += 1
                
            except queue.Empty:
                continue
                
    finally:
        print("Cleaning up...")
        csv_file.close()
        rgb_cam_center.destroy()
        rgb_cam_left.destroy()
        rgb_cam_right.destroy()
        seg_cam.destroy()
        lidar_cam.destroy()
        vehicle.destroy()
        for npc in npc_vehicles:
            if npc is not None:
                npc.destroy()
        pygame.quit()
        
    print(f"Data Collection Complete! Saved {frame_count} frames to {SAVE_DIR}")

if __name__ == '__main__':
    main()
