"""
Quick diagnostic: Load the trained model, feed it real images, 
and print what it actually predicts. This tells us if the model 
learned anything or is just predicting constant values.
"""
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from omni_models import OmniModel, OmniDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
model = OmniModel(seq_len=5).to(device)
model.load_state_dict(torch.load('omni_model.pth', map_location=device))
model.eval()

# Load dataset (same as training)
DATA_CSV = "/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Master/telemetry.csv"
IMAGE_DIR = "/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Master/images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = OmniDataset(csv_file=DATA_CSV, image_dir=IMAGE_DIR, seq_len=5, transform=transform)

print(f"\nDataset size: {len(dataset)} sequences")
print(f"{'='*70}")
print(f"{'IDX':>5} | {'ACTUAL_St':>10} {'ACTUAL_Th':>10} {'ACTUAL_Br':>10} | {'PRED_St':>10} {'PRED_Th':>10} {'PRED_Br':>10}")
print(f"{'='*70}")

# Test on 20 evenly spaced samples
test_indices = np.linspace(0, len(dataset)-1, 20, dtype=int)

all_pred_steer = []
all_pred_throttle = []
all_pred_brake = []

with torch.no_grad():
    for idx in test_indices:
        images, telemetry, commands, target = dataset[idx]
        
        # Add batch dimension
        images = images.unsqueeze(0).to(device)
        telemetry = telemetry.unsqueeze(0).to(device)
        commands = commands.unsqueeze(0).to(device)
        
        with torch.amp.autocast('cuda'):
            pred = model(images, telemetry, commands)
        
        ps, pt, pb = pred[0, 0].item(), pred[0, 1].item(), pred[0, 2].item()
        ts, tt, tb = target[0].item(), target[1].item(), target[2].item()
        
        all_pred_steer.append(ps)
        all_pred_throttle.append(pt)
        all_pred_brake.append(pb)
        
        print(f"{idx:5d} | {ts:10.4f} {tt:10.4f} {tb:10.4f} | {ps:10.4f} {pt:10.4f} {pb:10.4f}")

print(f"{'='*70}")
print(f"\n--- PREDICTION STATISTICS ---")
print(f"Steer    -> min: {min(all_pred_steer):.4f}, max: {max(all_pred_steer):.4f}, mean: {np.mean(all_pred_steer):.4f}")
print(f"Throttle -> min: {min(all_pred_throttle):.4f}, max: {max(all_pred_throttle):.4f}, mean: {np.mean(all_pred_throttle):.4f}")
print(f"Brake    -> min: {min(all_pred_brake):.4f}, max: {max(all_pred_brake):.4f}, mean: {np.mean(all_pred_brake):.4f}")

print(f"\n--- DIAGNOSIS ---")
steer_range = max(all_pred_steer) - min(all_pred_steer)
throttle_mean = np.mean(all_pred_throttle)
brake_mean = np.mean(all_pred_brake)

if steer_range < 0.1:
    print("WARNING: Steer predictions have almost NO variation. Model has NOT learned steering.")
else:
    print(f"OK: Steer has variation of {steer_range:.4f}")

if throttle_mean < 0.2:
    print(f"WARNING: Mean throttle is only {throttle_mean:.4f}. Model is barely accelerating.")
elif throttle_mean > 0.8:
    print(f"WARNING: Mean throttle is {throttle_mean:.4f}. Model is always flooring it.")
else:
    print(f"OK: Throttle mean is {throttle_mean:.4f}")

if brake_mean > 0.5:
    print(f"WARNING: Mean brake is {brake_mean:.4f}. Model thinks it should ALWAYS brake!")
else:
    print(f"OK: Brake mean is {brake_mean:.4f}")

if throttle_mean < brake_mean:
    print("\nCRITICAL: Model predicts MORE brake than throttle on average. Car will stay stuck!")
else:
    print(f"\nOK: Throttle ({throttle_mean:.4f}) > Brake ({brake_mean:.4f}) on average. Car should move.")
