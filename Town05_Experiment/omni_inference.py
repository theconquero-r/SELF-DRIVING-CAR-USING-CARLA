"""
Hybrid Self-Driving Car: Geometric Steering + Vision Perception
====================================================================
- Steering & Speed: CARLA BasicAgent (Flawless Math Control)
- Traffic Lights: Semantic Segmentation + HSV Detection
- Obstacles: Semantic Contour Distance Check
- Visuals: Full HUD with Road Mask and Bounding Boxes
"""
import carla
import pygame
import cv2
import math
import random
import numpy as np
import queue
import sys
import os
from ultralytics import YOLO

# Load our NEW CUSTOM trained YOLO model (Trained on 8660+ images!)
CUSTOM_MODEL_PATH = "/home/conqueror/AI_STORAGE/yolo_training_results/carla_yolov8s/weights/best.pt"
# Fallback to yolov8n.pt if training hasn't finished yet
if os.path.exists(CUSTOM_MODEL_PATH):
    YOLO_MODEL = YOLO(CUSTOM_MODEL_PATH)
    print(f"Loaded CUSTOM YOLO model from {CUSTOM_MODEL_PATH}")
else:
    YOLO_MODEL = YOLO(os.path.join(os.path.dirname(__file__), "yolov8n.pt"))
    print("Custom model not found, falling back to base yolov8n.pt")

carla_api_path = "/home/conqueror/AI_STORAGE/carla_packed_linux/CARLA_0.9.16/PythonAPI/carla"
if carla_api_path not in sys.path:
    sys.path.append(carla_api_path)
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

# --- Config ---
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FOV = 90.0
TARGET_SPEED = 25.0  # km/h - Safe city cruising speed

# ==========================================
# IMAGE PROCESSING
# ==========================================
def process_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    return array[:, :, :3].copy()

def get_traffic_light_state(frame, x1, y1, x2, y2):
    """Crops the traffic light box and checks for red or green pixels."""
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0: return "UNKNOWN"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Red masks (wraps around in HSV)
    mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = mask1 | mask2
    
    # Green mask
    green_mask = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))
    
    r_count = cv2.countNonZero(red_mask)
    g_count = cv2.countNonZero(green_mask)
    
    if r_count > g_count and r_count > 5:
        return "RED"
    elif g_count > r_count and g_count > 5:
        return "GREEN"
    return "UNKNOWN"

def build_master_image(rgb_frame):
    """Build master image using YOLOv8 for Vehicles (Step 1 of DL Pipeline)."""
    master = rgb_frame.copy()
    
    # MASK FP EGO HOOD:
    # Just in case the dashcam can see a bit of our own hood, we black out the bottom 15%
    yolo_input = rgb_frame.copy()
    cv2.rectangle(yolo_input, (0, int(IMG_HEIGHT * 0.85)), (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0), -1)
    
    # Run YOLO on the masked FP image
    results = YOLO_MODEL.predict(yolo_input, conf=0.25, imgsz=800, verbose=False)
    
    all_boxes = []
    overlay = master.copy()
    has_overlay = False
    
    tl_list = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        cx = (x1 + x2) / 2.0 / IMG_WIDTH
        
        if cls_id == 1:
            all_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
            cv2.putText(master, "VEHICLE", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            has_overlay = True
        elif cls_id == 0:
            all_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), -1)
            cv2.putText(master, "PEDESTRIAN", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            has_overlay = True
        elif cls_id == 3:
            cv2.rectangle(master, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(master, "SIGN", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        elif cls_id == 2:
            state = get_traffic_light_state(rgb_frame, x1, y1, x2, y2)
            tl_list.append({'state': state, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'dist': abs(cx - 0.5)})
            
    # SMART TRAFFIC LIGHT SELECTION
    tl_state = "NONE"
    if tl_list:
        # Pick the traffic light closest to the exact center of our view
        best_tl = min(tl_list, key=lambda x: x['dist'])
        for tl in tl_list:
            is_best = (tl == best_tl and tl['dist'] < 0.25)
            if is_best:
                tl_state = tl['state']
            
            color = (0,0,255) if tl['state']=="RED" else (0,255,0) if tl['state']=="GREEN" else (0,255,255)
            label = f"TL: {tl['state']}" if is_best else "TL: IGNORE"
            cv2.rectangle(master, (tl['x1'], tl['y1']), (tl['x2'], tl['y2']), color, 2 if is_best else 1)
            cv2.putText(master, label, (tl['x1'], max(0, tl['y1']-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if has_overlay:
        cv2.addWeighted(overlay, 0.4, master, 0.6, 0, master)
        
    return master, tl_state, all_boxes


def check_obstacle_ahead(vehicle_boxes, current_steer=0.0):
    """Check if any vehicle box is large and centered (= close and in our path)."""
    for (x1, y1, x2, y2) in vehicle_boxes:
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        cx = (x1 + x2) / 2.0 / IMG_WIDTH
        
        # SMART VISION CONE: 
        # If we are steering hard (turning), narrow the collision check to strictly the front 
        # so we don't accidentally brake for parked cars on the corner.
        if abs(current_steer) > 0.15:
            in_path = 0.42 < cx < 0.58  # Very tight center cone
        else:
            in_path = 0.40 < cx < 0.60  # Normal lane width
            
        # EMERGENCY BRAKE ZONE (FP CALIBRATED):
        # In First-Person view (FOV 90), a car 10 meters away has an area of ~12,000 pixels.
        # We only brake for vehicles that are very close (Area > 10000) and in our path.
        if box_area > 10000 and in_path:
            print(f"[DEBUG] OBSTACLE IN FRONT! Area={box_area}, cx={cx:.2f}, y2={y2}")
            return True
    return False

def spawn_pedestrians(world, bp_lib, count):
    """Spawn pedestrians with AI controllers."""
    walkers, controllers = [], []
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    ctrl_bp = bp_lib.find('controller.ai.walker')
    
    for _ in range(count):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        wp_bp = random.choice(walker_bps)
        if wp_bp.has_attribute('is_invincible'):
            wp_bp.set_attribute('is_invincible', 'false')
        walker = world.try_spawn_actor(wp_bp, carla.Transform(loc))
        if walker:
            walkers.append(walker)
    
    world.tick()
    for walker in walkers:
        ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), walker)
        controllers.append(ctrl)
    
    world.tick()
    for ctrl in controllers:
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(1.0 + random.random())
    
    return walkers, controllers


def main():
    print("=" * 60)
    print("  HYBRID Self-Driving Car (FINAL)")
    print("  Control: BasicAgent | Vision: Deep Learning")
    print("=" * 60)
    
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
    pygame.display.set_caption("AI Self-Driving Car (Geometric + Vision)")
    font = pygame.font.SysFont('Arial', 22, bold=True)
    
    # Connect to CARLA
    print("Connecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    # Cleanup
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.*'):
        a.destroy()
    
    # Spawn Ego
    ego_bp = bp_lib.filter('model3')[0]
    ego = world.spawn_actor(ego_bp, random.choice(spawn_points))
    
    # Spawn NPCs
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(10.0) # Normal traffic speed
    
    npc_bps = bp_lib.filter('vehicle.*')
    npcs = []
    
    # Spawn 15 random traffic vehicles across the map
    for _ in range(15):
        sp = random.choice(spawn_points)
        npc_bp = random.choice(npc_bps)
        npc = world.try_spawn_actor(npc_bp, sp)
        if npc:
            npc.set_autopilot(True)
            npcs.append(npc)
            
    print(f"Spawned {len(npcs)} NPC vehicles to create realistic traffic.")
    
    # Spawn 10 Pedestrians
    walkers, walker_ctrls = spawn_pedestrians(world, bp_lib, 10)
    print(f"Spawned {len(walkers)} pedestrians.")
    
    # ==========================
    # ==========================
    # SINGLE CAMERA SETUP (FIRST PERSON)
    # ==========================
    cam_t = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4), carla.Rotation(pitch=0.0))
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IMG_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
    rgb_bp.set_attribute('fov', str(FOV))
    rgb_cam = world.spawn_actor(rgb_bp, cam_t, attach_to=ego)
    rgb_q = queue.Queue()
    rgb_cam.listen(rgb_q.put)
    
    # Set up Geometric Logic Agent with tightly tuned PID for Model 3
    opt_dict = {
        'lateral_control_dict': {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05},
        'longitudinal_control_dict': {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05}
    }
    agent = BasicAgent(ego, target_speed=TARGET_SPEED, opt_dict=opt_dict)
    agent.ignore_traffic_lights(active=True)
    agent.ignore_stop_signs(active=True)
    agent.ignore_vehicles(active=True)
    agent.set_destination(random.choice(spawn_points).location)
    
    # Sync mode - Using 0.05 for very smooth BasicAgent control (20 FPS)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    spectator = world.get_spectator()
    
    print("Driving!")
    print("=" * 60)
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            world.tick()
            
            img = rgb_q.get(timeout=2.0)
            while not rgb_q.empty(): img = rgb_q.get()
            rgb_frame = process_rgb(img)
            
            # --- Perception ---
            master_image, tl_state, vehicle_boxes = build_master_image(rgb_frame)
            
            # --- Geometric Routing ---
            if agent.done():
                new_dest = random.choice(spawn_points).location
                agent.set_destination(new_dest)
                
            plan = agent.get_local_planner().get_plan()
            if plan and len(plan) > 3:
                wp_current = plan[0][0].transform.rotation.yaw
                wp_future = plan[min(len(plan)-1, 5)][0].transform.rotation.yaw
                angle_diff = (wp_future - wp_current + 180) % 360 - 180
                if abs(angle_diff) > 20.0:
                    agent.set_target_speed(12.0)
                else:
                    agent.set_target_speed(TARGET_SPEED)
            
            vel = ego.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # --- Update Spectator (Follow Car) ---
            t = ego.get_transform()
            spectator.set_transform(carla.Transform(
                t.location + carla.Location(z=15, x=-10),
                carla.Rotation(pitch=-30, yaw=t.rotation.yaw)
            ))
            
            control = agent.run_step()
            reason = "CRUISING"
            
            if tl_state == "RED" or tl_state == "YELLOW":
                control.throttle = 0.0
                control.brake = 1.0
                reason = "RED LIGHT"
            
            if check_obstacle_ahead(vehicle_boxes, control.steer):
                control.throttle = 0.0
                control.brake = 1.0
                reason = "OBSTACLE"
                
            print(f"[DEBUG] Reason: {reason:10s} | Spd: {speed_kmh:4.1f} km/h | Steer: {control.steer:5.2f}")
            ego.apply_control(control)
            
            # --- HUD DISPLAY (Single Screen) ---
            status_text = f"Spd: {int(speed_kmh)}/{int(TARGET_SPEED)} km/h | St: {control.steer:.2f} | Th: {control.throttle:.2f} | Br: {control.brake:.2f} | TL: {tl_state} | State: {reason}"
            cv2.rectangle(master_image, (0, 0), (800, 40), (30, 30, 30), -1)
            cv2.putText(master_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            disp = cv2.cvtColor(master_image, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(disp.swapaxes(0, 1))
            display.blit(surf, (0, 0))
            pygame.display.flip()
    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        rgb_cam.destroy()
        ego.destroy()
        
        for n in npcs:
            n.destroy()
            
        for ctrl in walker_ctrls:
            if ctrl.is_alive:
                ctrl.stop()
                ctrl.destroy()
        for w in walkers:
            if w.is_alive:
                w.destroy()
                
        pygame.quit()

if __name__ == "__main__":
    main()
