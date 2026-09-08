import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MultimodalDrivingModel(nn.Module):
    def __init__(self):
        super(MultimodalDrivingModel, self).__init__()
        
        # 1. Vision Encoder (MobileNetV3)
        # We use MobileNetV3 because it is incredibly fast and memory-efficient
        self.vision_backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 12 channels (3 stacked frames * 4 channels each)
        old_conv = self.vision_backbone.features[0][0]
        new_conv = nn.Conv2d(12, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        
        # Copy the pre-trained weights for the RGB channels across the 3 frames to retain ImageNet knowledge
        with torch.no_grad():
            for i in range(3):
                # RGB channels (0:3, 4:7, 8:11)
                new_conv.weight[:, i*4:i*4+3, :, :] = old_conv.weight / 3.0
                # Mask channels (3:4, 7:8, 11:12) initialized with mean of RGB weights
                new_conv.weight[:, i*4+3:i*4+4, :, :] = old_conv.weight.mean(dim=1, keepdim=True) / 3.0
            
        self.vision_backbone.features[0][0] = new_conv
        
        # Remove the classification head, we only need the feature extractor
        self.vision_out_features = self.vision_backbone.classifier[0].in_features # typically 576
        self.vision_backbone.classifier = nn.Identity()
        
        # 2. LiDAR BEV Encoder
        # Simple 2D CNN to compress the 128x128 Bird's Eye View map into a 128-dim vector
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU()
        )
        
        # 3. Metadata & YOLO Encoder
        # MLP to process Speed(1), Command(4), RedLight(1), VehicleDistance(1) = 7 inputs
        self.meta_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 4. Multimodal Fusion & Action Head
        fused_dim = self.vision_out_features + 128 + 64 # 576 + 128 + 64 = 768
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout to prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Outputs: Steer, Throttle, Brake
        )
        
        # 5. Vision Auxiliary Head (Multi-Task Learning)
        # Forces the CNN to learn depth and dynamics by predicting speed and vehicle distance purely from Vision!
        self.vision_aux_head = nn.Sequential(
            nn.Linear(self.vision_out_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs: Speed_Norm, Vehicle_Dist_Norm
        )
        
    def forward(self, vision, bev, meta):
        # Extract individual feature embeddings
        v_feat = self.vision_backbone(vision)
        l_feat = self.lidar_encoder(bev)
        m_feat = self.meta_encoder(meta)
        
        # Concatenate all embeddings into a single dense vector
        fused = torch.cat([v_feat, l_feat, m_feat], dim=1)
        
        # Predict raw actions
        actions = self.action_head(fused)
        
        # Predict Auxiliary tasks purely from vision
        aux_preds = self.vision_aux_head(v_feat)
        aux_speed = torch.sigmoid(aux_preds[:, 0:1])
        aux_dist = torch.sigmoid(aux_preds[:, 1:2])
        
        # Apply physical constraints to outputs
        steer = torch.tanh(actions[:, 0:1])       # Steer: -1.0 to 1.0
        throttle = torch.sigmoid(actions[:, 1:2]) # Throttle: 0.0 to 1.0
        brake = torch.sigmoid(actions[:, 2:3])    # Brake: 0.0 to 1.0
        
        return torch.cat([steer, throttle, brake], dim=1), torch.cat([aux_speed, aux_dist], dim=1)

if __name__ == '__main__':
    # Quick Test
    model = MultimodalDrivingModel()
    
    # Mock inputs matching dataset_loader.py output shapes (stacked frames)
    mock_vision = torch.randn(1, 12, 192, 320)
    mock_bev = torch.randn(1, 1, 128, 128)
    mock_meta = torch.randn(1, 7)
    
    out, aux_out = model(mock_vision, mock_bev, mock_meta)
    print("Model Test Success!")
    print(f"Action Output shape: {out.shape} (Expected: [1, 3])")
    print(f"Aux Output shape: {aux_out.shape} (Expected: [1, 2])")
    print(f"Sample Action Output: {out.detach().numpy()}")
