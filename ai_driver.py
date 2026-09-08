import carla
import pygame
import sys
import math
import numpy as np
import cv2
import time
import queue
import random
import sys
import os
import torch
from model import MultimodalDrivingModel

# REDIRECT EVERYTHING TO FILE FOR DEBUGGING
debug_file = open("debug_ai.log", "w", buffering=1)
sys.stdout = debug_file
sys.stderr = debug_file

# Add CARLA PythonAPI to sys.path so we can import agents
carla_api_path = "/home/conqueror/AI_STORAGE/carla_packed_linux/CARLA_0.9.16/PythonAPI/carla"
if carla_api_path not in sys.path:
    sys.path.append(carla_api_path)

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

# --- CONFIG ---
ONNX_MODEL_PATH = "best_model.onnx"
TARGET_TRAFFIC = 30
IMG_WIDTH = 1280
IMG_HEIGHT = 720
MODEL_IMG_SIZE = (320, 192)

# --- COLORS ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)

class CarlaUI:
    def __init__(self, world):
        pygame.init()
        self.display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
        pygame.display.set_caption("AI Driver - Map Selection")
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.world = world
        self.map = world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.waypoints = self.map.generate_waypoints(2.0)
        
        # Calculate map bounds for drawing using waypoints
        xs = [wp.transform.location.x for wp in self.waypoints]
        ys = [wp.transform.location.y for wp in self.waypoints]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        
        self.start_point = None
        self.end_point = None
        
    def map_to_pixel(self, location):
        # Scale CARLA coordinates to Pygame window
        padding = 50
        w = IMG_WIDTH - 2 * padding
        h = IMG_HEIGHT - 2 * padding
        
        x = padding + int((location.x - self.min_x) / (self.max_x - self.min_x) * w)
        y = padding + int((location.y - self.min_y) / (self.max_y - self.min_y) * h)
        return (x, y)
        
    def find_closest_waypoint(self, pos):
        w, h = self.display.get_width(), self.display.get_height()
        padding = 50
        min_dist = float('inf')
        closest_wp = None
        for wp in self.waypoints:
            x = padding + int((wp.transform.location.x - self.min_x) / (self.max_x - self.min_x) * (w - 2 * padding))
            y = padding + int((wp.transform.location.y - self.min_y) / (self.max_y - self.min_y) * (h - 2 * padding))
            dist = math.hypot(pos[0] - x, pos[1] - y)
            if dist < min_dist:
                min_dist = dist
                closest_wp = wp
        return closest_wp

    def select_route(self):
        print("Please select Start (Green) and End (Red) points on the map.")
        running = True
        while running:
            self.display.fill(BLACK)
            
            # Draw detailed road network (Google Maps 2D style)
            for wp in self.waypoints:
                px, py = self.map_to_pixel(wp.transform.location)
                pygame.draw.circle(self.display, (40, 40, 40), (px, py), 2)
                
            # Draw Start and End
            if self.start_point:
                pygame.draw.circle(self.display, GREEN, self.map_to_pixel(self.start_point.transform.location), 6)
            if self.end_point:
                pygame.draw.circle(self.display, RED, self.map_to_pixel(self.end_point.transform.location), 6)
                
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if not self.start_point:
                        wp = self.find_closest_waypoint(pos)
                        if wp: self.start_point = wp
                    elif not self.end_point:
                        wp = self.find_closest_waypoint(pos)
                        if wp: self.end_point = wp
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and self.start_point and self.end_point:
                        running = False
                        break
        
        # Elevate Z slightly to prevent ground collisions when spawning at waypoints
        start_transform = self.start_point.transform
        start_transform.location.z += 0.5
        end_transform = self.end_point.transform
        end_transform.location.z += 0.5
        return start_transform, end_transform

    def auto_select(self):
        # Auto select for headless debugging
        print("Auto-selecting points for debugging...")
        self.start_point = self.waypoints[0]
        self.end_point = self.waypoints[100]
        return self.start_point.transform, self.end_point.transform

# --- AI Data Processors ---
def process_lidar_to_bev(points, grid_size=128, range_m=25):
    bev = np.zeros((grid_size, grid_size), dtype=np.float32)
    mask = (np.abs(points[:, 0]) < range_m) & (np.abs(points[:, 1]) < range_m)
    points = points[mask]
    if len(points) == 0:
        return bev
    x_img = np.floor((points[:, 0] + range_m) / (2.0 * range_m) * grid_size).astype(np.int32)
    y_img = np.floor((points[:, 1] + range_m) / (2.0 * range_m) * grid_size).astype(np.int32)
    x_img = np.clip(x_img, 0, grid_size - 1)
    y_img = np.clip(y_img, 0, grid_size - 1)
    z_norm = np.clip((points[:, 2] + 2.5) / 5.0, 0, 1)
    bev[y_img, x_img] = z_norm
    return bev

def get_yolo_meta_optimized(ego_vehicle, vehicles, traffic_lights):
    # Returns the RAW distances to the nearest vehicle/red light in front
    red_light_dist = float('inf')
    vehicle_dist = float('inf')
    oncoming_dist = float('inf') # Track vehicles in the left/opposite lane
    
    ego_loc = ego_vehicle.get_location()
    ego_trans = ego_vehicle.get_transform()
    ego_fwd = ego_trans.get_forward_vector()
    ego_right = ego_trans.get_right_vector()
    
    # Check vehicles
    for actor in vehicles:
        if actor.id == ego_vehicle.id: continue
        dist = actor.get_location().distance(ego_loc)
        if dist < 60.0:
            dir_vec = actor.get_location() - ego_loc
            dot_fwd = ego_fwd.x * dir_vec.x + ego_fwd.y * dir_vec.y
            
            if dot_fwd > 0: # Vehicle is in front of us
                dot_right = ego_right.x * dir_vec.x + ego_right.y * dir_vec.y
                
                # If vehicle is strictly in our lane (+/- 2.0 meters laterally)
                if abs(dot_right) < 2.0:
                    if vehicle_dist == float('inf') or dist < vehicle_dist:
                        vehicle_dist = dist
                
                # If vehicle is in the adjacent left lane (oncoming or passing lane)
                elif dot_right < -2.0 and dot_right > -6.0:
                    if oncoming_dist == float('inf') or dist < oncoming_dist:
                        oncoming_dist = dist
                        
    # Check traffic lights
    for tl in traffic_lights:
        if tl.get_state() == carla.TrafficLightState.Red:
            dist = tl.get_location().distance(ego_loc)
            if dist < 50.0:
                dir_vec = tl.get_location() - ego_loc
                dot_fwd = ego_fwd.x * dir_vec.x + ego_fwd.y * dir_vec.y
                # Also check if traffic light is facing us
                tl_fwd = tl.get_transform().get_forward_vector()
                facing_dot = tl_fwd.x * dir_vec.x + tl_fwd.y * dir_vec.y
                if dot_fwd > 0 and facing_dot < 0:
                    if red_light_dist == float('inf') or dist < red_light_dist:
                        red_light_dist = dist
                        
    return red_light_dist, vehicle_dist, oncoming_dist

# --- MAIN LOOP ---
def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 1. UI Selection
    ui = CarlaUI(world)
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        start_sp, end_sp = ui.auto_select()
    else:
        start_sp, end_sp = ui.select_route()
    
    print(f"Start: {start_sp.location}, End: {end_sp.location}")
    
    # 2. Traffic Manager & Setup
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    # Spawn Traffic
    print(f"Spawning {TARGET_TRAFFIC} traffic vehicles...")
    traffic_actors = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for sp in spawn_points:
        if len(traffic_actors) >= TARGET_TRAFFIC: break
        if sp.location.distance(start_sp.location) < 10.0: continue # Don't spawn on ego
        bp = random.choice(blueprint_library.filter('vehicle.*'))
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, tm.get_port())
            traffic_actors.append(actor)
            
    # Spawn Ego
    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_vehicle = world.spawn_actor(ego_bp, start_sp)
    
    # Cache traffic lights once to avoid massive RPC lag during the loop
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    
    # 3. Route Planner
    print("Computing Route...")
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    route = grp.trace_route(start_sp.location, end_sp.location)
    route_waypoints = [wp for wp, cmd in route]
    route_commands = [cmd for wp, cmd in route]
    print(f"Route computed with {len(route)} waypoints!")
    
    # Sensors
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IMG_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
    rgb_bp.set_attribute('fov', '90')
    rgb_camera = world.spawn_actor(rgb_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego_vehicle)
    
    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', str(IMG_WIDTH))
    seg_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
    seg_bp.set_attribute('fov', '90')
    seg_camera = world.spawn_actor(seg_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego_vehicle)
    
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_camera = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego_vehicle)
    
    image_queue = queue.Queue()
    seg_queue = queue.Queue()
    lidar_queue = queue.Queue()
    
    rgb_camera.listen(image_queue.put)
    seg_camera.listen(seg_queue.put)
    lidar_camera.listen(lidar_queue.put)
    
    # PyTorch Model
    print("Loading PyTorch Model (CUDA)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalDrivingModel().to(device)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print("---------------------------------------------------------")
        print("WARNING: Old 'best_model.pth' is incompatible with the new 12-channel Frame Stacking architecture!")
        print("The AI will output random noise until you run 'python train.py' to retrain it.")
        print("Geometric Autopilot mode will still function perfectly.")
        print("---------------------------------------------------------")
    model.eval()
    
    command_map = {
        RoadOption.LEFT: [1, 0, 0, 0],
        RoadOption.RIGHT: [0, 1, 0, 0],
        RoadOption.STRAIGHT: [0, 0, 1, 0],
        RoadOption.LANEFOLLOW: [0, 0, 0, 1],
        RoadOption.CHANGELANELEFT: [1, 0, 0, 0],
        RoadOption.CHANGELANERIGHT: [0, 1, 0, 0],
        RoadOption.VOID: [0, 0, 0, 1],
    }
    
    print("Starting AI Drive!")
    use_ai_steer = False  # Toggle with SPACE
    
    frame_buffer = [] # Buffer to hold the last 3 vision tensors for Frame Stacking
    
    # DAgger recording setup
    dagger_dir = os.path.join('/home/conqueror/SELF_DRIVING_CAR/dataset', 'dagger_data')
    os.makedirs(dagger_dir, exist_ok=True)
    is_recording = False
    frame_count = 0
    
    is_overtaking = False
    overtake_timer = 0
    
    try:
        route_index = 0
        while True:
            # Process Pygame events to prevent OS freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        use_ai_steer = not use_ai_steer
                        print(f"Toggled AI Steer: {use_ai_steer}")
                    if event.key == pygame.K_r:
                        is_recording = not is_recording
                        print(f"DAgger Recording: {is_recording}")
                    
            world.tick()
            
            rgb_data = image_queue.get()
            seg_data = seg_queue.get()
            lidar_data = lidar_queue.get()
            
            # Sequential Route Progression (Prevents skipping corners/intersections)
            ego_loc = ego_vehicle.get_location()
            if route_index < len(route_waypoints) - 1:
                wp_loc = route_waypoints[route_index].transform.location
                # Pop waypoint if we are close to it
                if ego_loc.distance(wp_loc) < 3.0:
                    route_index += 1
                else:
                    # Fallback: if we overshot, check if the next waypoint is physically closer
                    next_wp_loc = route_waypoints[route_index + 1].transform.location
                    if ego_loc.distance(next_wp_loc) < ego_loc.distance(wp_loc):
                        route_index += 1
            
            if route_index >= len(route_waypoints) - 1:
                print("Destination Reached!")
                break
                
            current_cmd = route_commands[route_index]
            cmd_vec = command_map.get(current_cmd, [0, 0, 0, 1])
            
            # Process RGB
            img = np.frombuffer(rgb_data.raw_data, dtype=np.dtype("uint8")).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
            img = img[:, :, :3]
            display_img = img.copy() # For Pygame
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, MODEL_IMG_SIZE)
            img_tensor = img.astype(np.float32) / 255.0
            img_tensor = (img_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img_tensor = img_tensor.transpose(2, 0, 1) # (3, H, W)
            
            # Process Mask (MUST match training data exactly: 1=Road, 24=RoadLines)
            seg_img = np.frombuffer(seg_data.raw_data, dtype=np.dtype("uint8")).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
            seg_img = seg_img[:, :, 2] # Red channel contains class IDs
            mask = np.zeros_like(seg_img)
            mask[seg_img == 1] = 255 # Road
            mask[seg_img == 24] = 255 # Road line
            mask = cv2.resize(mask, MODEL_IMG_SIZE)
            mask_tensor = mask.astype(np.float32) / 255.0
            mask_tensor = np.expand_dims(mask_tensor, axis=0) # (1, H, W)
            
            vision_frame = np.concatenate([img_tensor, mask_tensor], axis=0)
            
            # --- Temporal Memory Buffer ---
            if len(frame_buffer) == 0:
                frame_buffer = [vision_frame, vision_frame, vision_frame] # Initialize with 3 copies
            else:
                frame_buffer.pop(0)
                frame_buffer.append(vision_frame)
                
            vision_tensor = np.concatenate(frame_buffer, axis=0) # (12, H, W)
            vision_tensor = np.expand_dims(vision_tensor, axis=0) # (1, 12, H, W)
            
            # Process LiDAR
            pts = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
            pts = np.reshape(pts, (int(pts.shape[0] / 4), 4))
            bev = process_lidar_to_bev(pts)
            bev_tensor = np.expand_dims(bev, axis=0)
            bev_tensor = np.expand_dims(bev_tensor, axis=0) # (1, 1, 128, 128)
            
            # Process Meta
            vel = ego_vehicle.get_velocity()
            speed_kmh = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            speed_norm = min(speed_kmh / 100.0, 1.0)
            
            # Pass traffic_actors (cached Python list) instead of calling world.get_actors()
            red_light_dist, vehicle_dist, oncoming_dist = get_yolo_meta_optimized(ego_vehicle, traffic_actors, traffic_lights)
            
            # Normalize for AI Model (1.0 = highly dangerous/close, 0.0 = safe/far)
            rl_norm = 1.0 - min(red_light_dist / 50.0, 1.0) if red_light_dist != float('inf') else 0.0
            veh_norm = 1.0 - min(vehicle_dist / 50.0, 1.0) if vehicle_dist != float('inf') else 0.0
            meta_tensor = np.array([[speed_norm] + cmd_vec + [rl_norm, veh_norm]], dtype=np.float32)
            
            # --- PyTorch Inference ---
            with torch.no_grad():
                v_in = torch.from_numpy(vision_tensor).float().to(device)
                b_in = torch.from_numpy(bev_tensor).float().to(device)
                m_in = torch.from_numpy(meta_tensor).float().to(device)
                
                outputs, aux_outputs = model(v_in, b_in, m_in)
                steer_ai, throttle_ai, brake_ai = outputs[0].cpu().numpy()
            
            # --- Geometric Controller Fallback ---
            target_wp_obj = None
            for i in range(route_index, min(route_index + 15, len(route_waypoints))):
                wp = route_waypoints[i]
                if ego_loc.distance(wp.transform.location) > 4.5:
                    target_wp_obj = wp
                    break
            if not target_wp_obj:
                target_wp_obj = route_waypoints[-1]
                
            target_wp_loc = target_wp_obj.transform.location
            
            # --- Geometric Speed & Brake Controller ---
            target_speed = 30.0 # Default cruising speed (km/h)
            hard_brake = False
            
            # Overtaking / Obstacle Avoidance Logic (Stateful to prevent oscillation)
            if is_overtaking:
                overtake_timer -= 1
                target_speed = 20.0
                # Keep target waypoint shifted left
                wp_right = target_wp_obj.transform.get_right_vector()
                target_wp_loc.x -= wp_right.x * 3.5
                target_wp_loc.y -= wp_right.y * 3.5
                if overtake_timer <= 0:
                    is_overtaking = False
                    
            elif vehicle_dist < 20.0:
                if oncoming_dist > 40.0:
                    # LEFT LANE IS CLEAR! Trigger overtake state!
                    is_overtaking = True
                    overtake_timer = 100 # ~5 seconds at 20fps
                    target_speed = 20.0
                    wp_right = target_wp_obj.transform.get_right_vector()
                    target_wp_loc.x -= wp_right.x * 3.5
                    target_wp_loc.y -= wp_right.y * 3.5
                else:
                    # Left lane is BLOCKED by oncoming traffic. We MUST stop and wait!
                    if vehicle_dist < 8.0:
                        target_speed = 0.0
                        hard_brake = True
                    else:
                        target_speed = 5.0
            
            # Traffic Light Logic overrides everything
            if red_light_dist < 12.0:
                target_speed = 0.0
                hard_brake = True
            elif red_light_dist < 25.0:
                target_speed = min(target_speed, 10.0)
                
            # Emergency crash prevention
            if vehicle_dist < 6.0:
                target_speed = 0.0
                hard_brake = True
                
            ego_transform = ego_vehicle.get_transform()
            ego_fwd = ego_transform.get_forward_vector()
            vec_target = target_wp_loc - ego_loc
            vec_norm = math.hypot(vec_target.x, vec_target.y)
            if vec_norm > 0:
                vec_target.x /= vec_norm
                vec_target.y /= vec_norm
            # Cross product determines Left/Right steer
            cross_prod = ego_fwd.x * vec_target.y - ego_fwd.y * vec_target.x
            steer_geom = float(np.clip(cross_prod * 2.5, -1.0, 1.0))
            
            # Slow down on sharp turns
            if abs(steer_geom) > 0.2:
                target_speed = min(target_speed, 15.0) 
                
            speed_error = target_speed - speed_kmh
            if speed_error > 0 and not hard_brake:
                throttle_geom = float(np.clip(speed_error / 15.0, 0.0, 0.6))
                brake_geom = 0.0
            else:
                throttle_geom = 0.0
                brake_geom = float(np.clip(-speed_error / 10.0, 0.0, 1.0))
                if hard_brake: brake_geom = 1.0
            
            # --- Final Control Selection ---
            if use_ai_steer:
                # Deep Learning Model Control
                steer_sign = np.sign(steer_ai)
                boosted_steer = steer_sign * (abs(steer_ai) ** 0.6)
                steer = float(np.clip(boosted_steer * 1.5, -1.0, 1.0))
                throttle = float(np.clip(throttle_ai, 0.0, 1.0))
                brake = float(np.clip(brake_ai, 0.0, 1.0))
            else:
                # Full-Proof Geometric Controller (Custom Algorithm)
                steer = steer_geom
                throttle = throttle_geom
                brake = brake_geom
            
            # Remove noise
            if abs(steer) < 0.02: steer = 0.0
            
            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = throttle
            control.brake = brake
            control.manual_gear_shift = False
            control.gear = 1
            ego_vehicle.apply_control(control)
            if is_recording:
                # Save data for retraining the AI with perfect geometric actions
                frame_name = f"dagger_{frame_count:05d}"
                # Save Image
                cv2.imwrite(os.path.join(dagger_dir, f"{frame_name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # Append to CSV
                csv_path = os.path.join(dagger_dir, 'dagger_data.csv')
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w') as f:
                        f.write("Image_Name,Speed,Command,Steer,Throttle,Brake,YOLO_JSON\n")
                with open(csv_path, 'a') as f:
                    f.write(f"{frame_name},{speed_kmh},{current_cmd.name},{steer_geom},{throttle_geom},{brake_geom},{{}}\n")
                frame_count += 1
            
            # --- Update UI ---
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(display_img.swapaxes(0, 1))
            ui.display.blit(surface, (0, 0))
            
            # HUD
            text_speed = ui.font.render(f"Speed: {speed_kmh:.1f} km/h", True, GREEN)
            text_cmd = ui.font.render(f"Cmd: {current_cmd.name}", True, GREEN)
            steer_mode = "AI Model" if use_ai_steer else "Geometric"
            text_mode = ui.font.render(f"Steering: {steer_mode} (Press SPACE to toggle)", True, GREEN if use_ai_steer else WHITE)
            text_rec = ui.font.render(f"REC (Press R)", True, RED if is_recording else GRAY)
            
            text_steer = ui.font.render(f"Steer: {steer:.2f}", True, WHITE)
            text_th = ui.font.render(f"Throttle: {throttle:.2f}", True, WHITE)
            text_br = ui.font.render(f"Brake: {brake:.2f}", True, WHITE)
            
            ui.display.blit(text_speed, (20, 20))
            ui.display.blit(text_cmd, (20, 50))
            ui.display.blit(text_mode, (20, 80))
            ui.display.blit(text_rec, (20, 110))
            
            ui.display.blit(text_steer, (IMG_WIDTH - 150, 20))
            ui.display.blit(text_th, (IMG_WIDTH - 150, 50))
            ui.display.blit(text_br, (IMG_WIDTH - 150, 80))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                    
    except KeyboardInterrupt:
        print("Stopped by User.")
    except Exception as e:
        import traceback
        with open("ai_crash.log", "w") as f:
            traceback.print_exc(file=f)
        print(f"CRASHED: {e}. Check ai_crash.log")
    finally:
        print("Destroying actors...")
        if 'settings' in locals():
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if 'rgb_camera' in locals() and rgb_camera is not None: rgb_camera.destroy()
        if 'seg_camera' in locals() and seg_camera is not None: seg_camera.destroy()
        if 'lidar_camera' in locals() and lidar_camera is not None: lidar_camera.destroy()
        if 'ego_vehicle' in locals() and ego_vehicle is not None: ego_vehicle.destroy()
        if 'traffic_actors' in locals(): client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
        pygame.quit()

if __name__ == '__main__':
    main()
