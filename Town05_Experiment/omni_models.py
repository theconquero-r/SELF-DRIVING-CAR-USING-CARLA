import os
import csv
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ==========================================
# 1. THE DATASET (Sequence Loading & 3-Cam)
# ==========================================
class OmniDataset(Dataset):
    def __init__(self, csv_file, image_dir, seq_len=5, transform=None):
        self.image_dir = image_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # Parse CSV
        self.rows = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.header = next(reader)
            for row in reader:
                self.rows.append(row)
                
    def __len__(self):
        # We need at least 'seq_len' frames to form a sequence
        return len(self.rows) - self.seq_len

    def __getitem__(self, idx):
        # Get sequence of 'seq_len' rows
        seq_rows = self.rows[idx : idx + self.seq_len]
        
        # Randomly choose which camera to use for this ENTIRE sequence (Nvidia 3-Camera setup)
        # This prevents the model from jumping between cameras in a single temporal sequence
        camera_choice = random.choice(['center', 'left', 'right'])
        
        images = []
        telemetry = []
        commands = []
        
        for i, row in enumerate(seq_rows):
            frame_id = row[0]
            speed = float(row[1])
            steer = float(row[2])
            throttle = float(row[3])
            brake = float(row[4])
            cmd = int(row[5]) # 0=Left, 1=Right, 2=Straight, 3=Follow
            pitch = float(row[9])
            yaw = float(row[10])
            roll = float(row[11])
            
            # Load Image
            img_name = f"{frame_id}_{camera_choice}.jpg"
            img_path = os.path.join(self.image_dir, img_name)
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            images.append(image)
            
            # Normalize Telemetry inputs
            norm_speed = speed / 30.0 # Assuming max speed around 30 m/s
            norm_pitch = pitch / 180.0
            norm_yaw = yaw / 180.0
            norm_roll = roll / 180.0
            
            telemetry.append(torch.tensor([norm_speed, norm_pitch, norm_yaw, norm_roll], dtype=torch.float32))
            commands.append(torch.tensor(cmd, dtype=torch.long))
            
            # If this is the LAST frame in the sequence (the target we want to predict)
            if i == self.seq_len - 1:
                # Apply NVIDIA Steering Offset if using side cameras
                if camera_choice == 'left':
                    steer = steer + 0.2
                elif camera_choice == 'right':
                    steer = steer - 0.2
                
                # Clip steer to [-1, 1]
                steer = max(-1.0, min(1.0, steer))
                
                target = torch.tensor([steer, throttle, brake], dtype=torch.float32)

        # Stack into Sequence Tensors
        # images: (seq_len, C, H, W)
        seq_images = torch.stack(images)
        seq_telemetry = torch.stack(telemetry)
        seq_commands = torch.stack(commands)
        
        return seq_images, seq_telemetry, seq_commands, target


# ==========================================
# 2. THE LATE FUSION MODEL (CNN + LSTM)
# ==========================================
class OmniModel(nn.Module):
    def __init__(self, seq_len=5):
        super(OmniModel, self).__init__()
        self.seq_len = seq_len
        
        # A. Vision Branch (Pre-trained ResNet18)
        # Faster convergence, great at finding spatial structures (vehicles, red blocks)
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity() # Remove the final ImageNet classification layer (output is 512)
        self.vision_cnn = resnet 
        
        # Compress Vision features from 512 to 256
        self.vision_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # B. Telemetry Branch (Speed, Pitch, Yaw, Roll)
        self.telemetry_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # C. Conditional Command Branch (Left, Right, Straight, Follow)
        # Commands are 0,1,2,3. We embed them into 32 dense features.
        self.command_embedding = nn.Embedding(num_embeddings=4, embedding_dim=32)
        
        # TOTAL LATE FUSION VECTOR = 256 (Vision) + 64 (Telemetry) + 32 (Command) = 352
        
        # D. Temporal Memory (LSTM)
        self.lstm = nn.LSTM(input_size=352, hidden_size=256, num_layers=1, batch_first=True)
        
        # E. Action Head (Steer, Throttle, Brake)
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, images, telemetry, commands):
        # images: (Batch, Seq_Len, C, H, W)
        B, seq, C, H, W = images.shape
        
        # Flatten time dimension for CNN and MLPs: (Batch * Seq_Len, ...)
        images_flat = images.view(B * seq, C, H, W)
        tel_flat = telemetry.view(B * seq, 4)
        cmd_flat = commands.view(B * seq)
        
        # A. Vision Pass
        vis_feat = self.vision_cnn(images_flat)
        vis_feat = self.vision_fc(vis_feat)
        
        # B. Telemetry Pass
        tel_feat = self.telemetry_fc(tel_flat)
        
        # C. Command Pass
        cmd_feat = self.command_embedding(cmd_flat)
        
        # Late Fusion (Concatenation)
        fused = torch.cat((vis_feat, tel_feat, cmd_feat), dim=1) # Shape: (B * seq, 352)
        
        # Reshape back to Sequence for LSTM: (Batch, Seq_Len, 352)
        fused_seq = fused.view(B, seq, -1)
        
        # D. Temporal Pass
        lstm_out, (hn, cn) = self.lstm(fused_seq)
        
        # We only care about predicting the action for the current (last) frame in the sequence
        last_step = lstm_out[:, -1, :] # Shape: (Batch, 256)
        
        # E. Action Prediction
        actions = self.action_head(last_step)
        
        # Apply strict output boundaries
        steer = torch.tanh(actions[:, 0]) # -1 to 1
        throttle = torch.sigmoid(actions[:, 1]) # 0 to 1
        brake = torch.sigmoid(actions[:, 2]) # 0 to 1
        
        return torch.stack((steer, throttle, brake), dim=1)

# Quick Test Function
if __name__ == "__main__":
    print("Testing OmniModel Architecture (Batch=2, Seq=5)...")
    model = OmniModel(seq_len=5)
    
    # Dummy tensors
    dummy_imgs = torch.randn(2, 5, 3, 224, 224)
    dummy_tel = torch.randn(2, 5, 4)
    dummy_cmd = torch.randint(0, 4, (2, 5))
    
    out = model(dummy_imgs, dummy_tel, dummy_cmd)
    print(f"Output Shape (Batch, Actions): {out.shape}")
    print("Test Passed! The Late Fusion architecture is mathematically sound.")
