import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from omni_models import OmniDataset, OmniModel
from tqdm import tqdm

def main():
    # 1. Hyperparameters (Optimized for Deep Learning Convergence)
    BATCH_SIZE = 8
    SEQ_LEN = 5
    EPOCHS = 30 # Full training phase for precise lane keeping
    LEARNING_RATE = 1e-4
    
    DATA_CSV = "/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Master/telemetry.csv"
    IMAGE_DIR = "/home/conqueror/AI_STORAGE/SELF_DRIVING_DATASETS/Town05_Master/images"
    MODEL_SAVE_PATH = "omni_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data Preparation
    # ResNet expects 224x224, but our images are likely larger/different. We resize them.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization since we use pretrained ResNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading Dataset...")
    dataset = OmniDataset(csv_file=DATA_CSV, image_dir=IMAGE_DIR, seq_len=SEQ_LEN, transform=transform)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # 3. Model & Optimizer
    model = OmniModel(seq_len=SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Loss Function: MSE
    criterion = nn.MSELoss()
    
    # Mixed Precision Scaler for Fast & Low-Memory Training
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]")
        for images, telemetry, commands, targets in loop:
            images = images.to(device)
            telemetry = telemetry.to(device)
            commands = commands.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision (AMP)
            with torch.cuda.amp.autocast():
                predictions = model(images, telemetry, commands)
                
                loss_steer = criterion(predictions[:, 0], targets[:, 0])
                loss_throttle = criterion(predictions[:, 1], targets[:, 1])
                loss_brake = criterion(predictions[:, 2], targets[:, 2])
                
                total_loss = (2.0 * loss_steer) + (1.0 * loss_throttle) + (1.0 * loss_brake)
            
            # Scaled Backward Pass
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        
        # Use AMP in validation too for speed
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]")
                for images, telemetry, commands, targets in loop:
                    images = images.to(device)
                    telemetry = telemetry.to(device)
                    commands = commands.to(device)
                    targets = targets.to(device)
                    
                    predictions = model(images, telemetry, commands)
                    
                    loss_steer = criterion(predictions[:, 0], targets[:, 0])
                    loss_throttle = criterion(predictions[:, 1], targets[:, 1])
                    loss_brake = criterion(predictions[:, 2], targets[:, 2])
                    total_loss = (2.0 * loss_steer) + (1.0 * loss_throttle) + (1.0 * loss_brake)
                    
                    val_loss += total_loss.item()
                    loop.set_postfix(val_loss=total_loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[*] New Best Model Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
