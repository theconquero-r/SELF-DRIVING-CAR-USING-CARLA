import cv2
import numpy as np
import os
import glob

# CARLA CityScapes Palette Colors (BGR format for OpenCV)
# Vehicles: Dark Blue -> RGB (142, 0, 0) -> BGR (0, 0, 142)
COLOR_VEHICLE = (0, 0, 142) 
# Pedestrians: Red -> RGB (220, 20, 60) -> BGR (60, 20, 220)
COLOR_PEDESTRIAN = (60, 20, 220)
# Traffic Signals: Orange -> RGB (250, 170, 30) -> BGR (30, 170, 250)
COLOR_LIGHT = (30, 170, 250)
# Traffic Signs: Yellow -> RGB (220, 220, 0) -> BGR (0, 220, 220)
COLOR_SIGN = (0, 220, 220)

# YOLO Class IDs
CLASS_VEHICLE = 0
CLASS_PEDESTRIAN = 1
CLASS_LIGHT = 2
CLASS_SIGN = 3

def mask_to_yolo():
    os.makedirs("dataset/labels", exist_ok=True)
    mask_files = sorted(glob.glob("dataset/masks/frame_*.png"))
    
    print(f"Processing ALL {len(mask_files)} frames for Auto-Labeling...")
    
    processed_count = 0
    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        frame_id = filename.split('.')[0]
        
        mask_img = cv2.imread(mask_path)
        if mask_img is None:
            continue
            
        height, width, _ = mask_img.shape
        yolo_labels = []
        
        def extract_boxes(color, class_id):
            lower = np.array(color)
            upper = np.array(color)
            binary_mask = cv2.inRange(mask_img, lower, upper)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter out tiny noise (e.g. less than 10x10 pixels)
                if w * h < 100:
                    continue
                    
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                
        # Extract bounding boxes for all 4 important classes!
        extract_boxes(COLOR_VEHICLE, CLASS_VEHICLE)
        extract_boxes(COLOR_PEDESTRIAN, CLASS_PEDESTRIAN)
        extract_boxes(COLOR_LIGHT, CLASS_LIGHT)
        extract_boxes(COLOR_SIGN, CLASS_SIGN)
        
        txt_path = f"dataset/labels/{frame_id}.txt"
        with open(txt_path, 'w') as f:
            f.write("\n".join(yolo_labels))
            
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Labeled {processed_count} / {len(mask_files)} frames...")
            
    print(f"✅ Success! Master Dataset Auto-Labeled {processed_count} frames.")
    print("Labels saved in 'dataset/labels/' ready for YOLO fine-tuning!")

if __name__ == '__main__':
    mask_to_yolo()
