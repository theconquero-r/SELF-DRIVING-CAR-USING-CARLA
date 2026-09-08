import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarlaMultimodalDataset(Dataset):
    def __init__(self, csv_file, dataset_dir, img_size=(320, 192), transform=None, is_train=True):
        """
        Args:
            csv_file (string): Path to the action_data.csv file.
            dataset_dir (string): Directory with all the data folders.
            img_size (tuple): Desired resolution for RGB and Mask (width, height) - multiple of 32.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        
        # Load the massive CSV efficiently
        print(f"Loading Dataset from {csv_file}...")
        self.data_frame = pd.read_csv(csv_file, header=0, low_memory=False)
        self.data_frame.rename(columns={'Speed_kmh': 'Speed'}, inplace=True)
        
        # Data balancing (To prevent AI from just driving straight forever)
        # We drop 70% of straight driving frames during training phase
        if is_train:
            print(f"Total samples before balancing: {len(self.data_frame)}")
            straight_mask = (abs(self.data_frame['Steer']) < 0.05)
            turn_mask = ~straight_mask
            
            # Keep all turns, drop a fraction of straight
            straight_data = self.data_frame[straight_mask]
            turn_data = self.data_frame[turn_mask]
            
            # Sample only 30% of straight data
            straight_data_sampled = straight_data.sample(frac=0.3, random_state=42)
            self.data_frame = pd.concat([turn_data, straight_data_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f"Total samples after balancing (focusing on turns): {len(self.data_frame)}")
        
        # Command Mapping (One-hot encoding)
        self.command_map = {
            'Left': [1, 0, 0, 0],
            'Right': [0, 1, 0, 0],
            'Straight': [0, 0, 1, 0],
            'LaneFollow': [0, 0, 0, 1]
        }
        
        # Image transforms
        # Image transforms
        if is_train:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_frame)

    def process_lidar_to_bev(self, points, grid_size=128, range_m=25):
        """
        Fast 2D BEV projection.
        Grid size 128x128 representing 50m x 50m around the car.
        """
        bev = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Filter points within range
        mask = (np.abs(points[:, 0]) < range_m) & (np.abs(points[:, 1]) < range_m)
        points = points[mask]
        
        if len(points) == 0:
            return np.expand_dims(bev, axis=0)
            
        # Map X, Y to pixel coordinates
        x_img = np.floor((points[:, 0] + range_m) / (2.0 * range_m) * grid_size).astype(np.int32)
        y_img = np.floor((points[:, 1] + range_m) / (2.0 * range_m) * grid_size).astype(np.int32)
        
        # Clip to bounds
        x_img = np.clip(x_img, 0, grid_size - 1)
        y_img = np.clip(y_img, 0, grid_size - 1)
        
        # Normalize height (Z) between 0 and 1 roughly. Z is usually -2.5m (road) to +3m (trucks).
        z_norm = np.clip((points[:, 2] + 2.5) / 5.0, 0, 1)
        
        # We take the maximum height per pixel (to see tall obstacles)
        bev[y_img, x_img] = z_norm
        
        return np.expand_dims(bev, axis=0) # Shape: (1, 128, 128)

    def extract_yolo_metadata(self, yolo_str):
        """ Extract binary presence of critical objects like Red Traffic Lights. """
        red_light_dist = 0.0
        vehicle_dist = 0.0
        
        try:
            objs = json.loads(yolo_str)
            for obj in objs:
                if obj['class'] == 'traffic_light' and obj['state'] == 'Red':
                    dist = obj['depth_m']
                    if red_light_dist == 0.0 or dist < red_light_dist:
                        red_light_dist = dist
                elif obj['class'] == 'vehicle':
                    dist = obj['depth_m']
                    if vehicle_dist == 0.0 or dist < vehicle_dist:
                        vehicle_dist = dist
        except:
            pass
            
        # Normalize (cap at 50m)
        red_light = 1.0 - min(red_light_dist / 50.0, 1.0) if red_light_dist > 0 else 0.0
        vehicle = 1.0 - min(vehicle_dist / 50.0, 1.0) if vehicle_dist > 0 else 0.0
        
        return [red_light, vehicle]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        base_name = row['Image_Name']
        
        # --- TEMPORAL MEMORY (Frame Stacking) ---
        # Instead of an LSTM, we stack the last 3 frames (t, t-1, t-2) to give the model 'Memory'.
        # Since base_name is like 'frame_0015', we can dynamically load previous frames!
        try:
            current_idx = int(base_name.split('_')[1])
        except:
            current_idx = 0
            
        stacked_vision = []
        for t_offset in [0, -1, -2]: # Current, previous, previous-previous
            t_idx = max(0, current_idx + t_offset)
            t_name = f"frame_{t_idx:04d}"
            
            # Load RGB
            img_path = os.path.join(self.dataset_dir, 'yolo_data', 'images', f"{t_name}.jpg")
            if not os.path.exists(img_path):
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            img_tensor = self.img_transform(image)
            
            # Load Mask
            mask_path = os.path.join(self.dataset_dir, 'segmentation_data', 'masks', f"{t_name}.png")
            if not os.path.exists(mask_path):
                mask = np.zeros((720, 1280), dtype=np.uint8)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            
            # Cat to (4, H, W)
            v_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
            stacked_vision.append(v_tensor)
            
        # Concatenate along channel dim: (12, H, W)
        vision_tensor = torch.cat(stacked_vision, dim=0)
        
        # 3. Load LiDAR
        lidar_path = os.path.join(self.dataset_dir, 'lidar_data', f"{base_name}.npz")
        try:
            lidar_data = np.load(lidar_path)['points']
            bev_tensor = torch.from_numpy(self.process_lidar_to_bev(lidar_data))
        except:
            bev_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)
            
        # 4. Metadata Vector
        speed = min(float(row['Speed']) / 100.0, 1.0) # Normalize speed 0-100 km/h -> 0-1
        cmd_str = row['Command']
        cmd_vec = self.command_map.get(cmd_str, [0, 0, 0, 1])
        yolo_vec = self.extract_yolo_metadata(row['YOLO_JSON'])
        
        meta_tensor = torch.tensor([speed] + cmd_vec + yolo_vec, dtype=torch.float32)
        
        # 5. Targets (Actions)
        throttle = float(row['Throttle'])
        steer = float(row['Steer'])
        brake = float(row['Brake'])
        action_tensor = torch.tensor([steer, throttle, brake], dtype=torch.float32)
        
        return {
            'vision': vision_tensor,
            'bev': bev_tensor,
            'meta': meta_tensor,
            'actions': action_tensor
        }

if __name__ == '__main__':
    # Quick Test
    dataset = CarlaMultimodalDataset(
        csv_file='/home/conqueror/SELF_DRIVING_CAR/dataset/action_model_data/action_data.csv',
        dataset_dir='/home/conqueror/SELF_DRIVING_CAR/dataset',
        is_train=False
    )
    print("Dataset Test Success!")
    print(f"Sample 0 output keys: {dataset[0].keys()}")
    print(f"Vision shape: {dataset[0]['vision'].shape}")
    print(f"BEV shape: {dataset[0]['bev'].shape}")
    print(f"Meta shape: {dataset[0]['meta'].shape}")
    print(f"Actions shape: {dataset[0]['actions'].shape}")
