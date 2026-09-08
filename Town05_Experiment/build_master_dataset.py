import os
import csv
import cv2
import json
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

DATA_DIR = "/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Omni"
OUT_DIR = "/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Master"

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

def process_single_image(rgb_path, mask_img, yolo_json, tl_state, out_path):
    if not os.path.exists(rgb_path): return False
    
    # Load RGB
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None: return False
    
    h, w = rgb_img.shape[:2]
    
    # 1. Overlay Road Mask (Light Green Transparent)
    if mask_img is not None:
        mask_color = np.zeros_like(rgb_img)
        mask_color[mask_img == 255] = [0, 255, 0] # Green in BGR
        
        # Blend the mask with original image (30% opacity)
        alpha = 0.3
        mask_area = mask_img == 255
        rgb_img[mask_area] = cv2.addWeighted(rgb_img, 1-alpha, mask_color, alpha, 0)[mask_area]
        
    # 2. Draw YOLO Vehicles (Solid Blue Blocks)
    for obj in yolo_json:
        if obj['class'] == 'vehicle' or obj['class'] == 'pedestrian':
            cx, cy, bw, bh = obj['box']
            x_min = int((cx - bw/2) * w)
            y_min = int((cy - bh/2) * h)
            x_max = int((cx + bw/2) * w)
            y_max = int((cy + bh/2) * h)
            
            # Draw semi-transparent blue block for vehicles
            overlay = rgb_img.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), -1) # Blue
            cv2.addWeighted(overlay, 0.4, rgb_img, 0.6, 0, rgb_img)

    # 3. Ground Truth Logic for Ego-Lane Traffic Light
    # We use the exact 'tl_state' from the CARLA API for the color.
    # To find the CORRECT bounding box, we look at the right half of the screen (cx > 0.5) 
    # because opposite-lane lights are on the left. Background trees can cause false colors in JSON, so we ignore JSON color.
    if tl_state != 'NONE':
        valid_lights = []
        for obj in yolo_json:
            if obj['class'] == 'traffic_light':
                cx, cy, bw, bh = obj['box']
                if cx > 0.45: # Ego traffic lights are on the right/center
                    area = bw * bh
                    valid_lights.append((area, obj))
        
        # If we found lights matching our ego-state, pick the largest one (closest to us)
        if valid_lights:
            valid_lights.sort(key=lambda x: x[0], reverse=True)
            best_light = valid_lights[0][1]
            
            cx, cy, bw, bh = best_light['box']
            
            # Inflate box by 50% to make it extremely prominent for CNN
            bw *= 1.5
            bh *= 1.5
            
            x_min = int((cx - bw/2) * w)
            y_min = int((cy - bh/2) * h)
            x_max = int((cx + bw/2) * w)
            y_max = int((cy + bh/2) * h)
            
            if tl_state == 'RED': block_color = (0, 0, 255)
            elif tl_state == 'GREEN': block_color = (0, 255, 0)
            elif tl_state == 'YELLOW': block_color = (0, 255, 255)
            else: block_color = (255, 255, 255)
            
            # Draw MASSIVE SOLID BLOCK (100% opaque)
            cv2.rectangle(rgb_img, (x_min, y_min), (x_max, y_max), block_color, -1)
        
    cv2.imwrite(out_path, rgb_img)
    return True

def process_row(row):
    frame_id, speed, steer, throttle, brake, cmd, gx, gy, gz, pitch, yaw, roll, tl_state, yolo_str = row
    
    yolo_json = json.loads(yolo_str)
    
    mask_path = os.path.join(DATA_DIR, "masks", f"{frame_id}.png")
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Process Center
    process_single_image(
        os.path.join(DATA_DIR, "images", f"{frame_id}_center.jpg"),
        mask_img, yolo_json, tl_state,
        os.path.join(OUT_DIR, "images", f"{frame_id}_center.jpg")
    )
    # Process Left
    process_single_image(
        os.path.join(DATA_DIR, "images", f"{frame_id}_left.jpg"),
        mask_img, yolo_json, tl_state,
        os.path.join(OUT_DIR, "images", f"{frame_id}_left.jpg")
    )
    # Process Right
    process_single_image(
        os.path.join(DATA_DIR, "images", f"{frame_id}_right.jpg"),
        mask_img, yolo_json, tl_state,
        os.path.join(OUT_DIR, "images", f"{frame_id}_right.jpg")
    )
    
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help="Number of frames to process (0 for all)")
    args = parser.parse_args()
    
    in_csv = os.path.join(DATA_DIR, "telemetry.csv")
    out_csv = os.path.join(OUT_DIR, "telemetry.csv")
    
    with open(in_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        
    if args.limit > 0:
        rows = rows[:args.limit]
        
    print(f"Starting Master Image Pipeline on {len(rows)} frames...")
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Use multiprocessing to speed up the massive dataset processing
        with Pool(cpu_count()) as pool:
            for processed_row in tqdm(pool.imap(process_row, rows), total=len(rows)):
                writer.writerow(processed_row)

    print("Phase 2: Master Dataset processing complete!")

if __name__ == "__main__":
    main()
