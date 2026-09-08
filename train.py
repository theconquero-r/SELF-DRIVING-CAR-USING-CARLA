import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_loader import CarlaMultimodalDataset
from model import MultimodalDrivingModel

def train():
    # --- Configurations ---
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_CSV = '/home/conqueror/SELF_DRIVING_CAR/dataset/action_model_data/action_data.csv'
    DATASET_DIR = '/home/conqueror/SELF_DRIVING_CAR/dataset'
    
    print(f"Starting Training on DEVICE: {DEVICE}")
    
    # --- Setup Dataset & DataLoader ---
    # We will use 90% for training and 10% for validation
    full_dataset = CarlaMultimodalDataset(csv_file=DATASET_CSV, dataset_dir=DATASET_DIR, is_train=True)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # num_workers=4 for fast data loading on CPU while GPU trains
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Setup Model, Loss, Optimizer ---
    model = MultimodalDrivingModel().to(DEVICE)
    
    # --- Custom Weighted Loss Function ---
    # We heavily penalize errors when the target steering or braking is high.
    # This completely solves the "Regression to the Mean" issue in Behavioral Cloning!
    def weighted_mse_loss(pred_actions, target_actions):
        mse = (pred_actions - target_actions) ** 2
        
        # weights shape: [batch_size, 3] (Steer, Throttle, Brake)
        steer_weights = 1.0 + 10.0 * (torch.abs(target_actions[:, 0]) > 0.05).float()
        brake_weights = 1.0 + 5.0 * (target_actions[:, 2] > 0.1).float()
        
        weights = torch.stack([steer_weights, torch.ones_like(steer_weights), brake_weights], dim=1)
        return torch.mean(mse * weights)
        
    aux_criterion = nn.MSELoss() # Standard MSE for Auxiliary tasks
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # --- Automatic Mixed Precision (AMP) ---
    # This halves VRAM usage and speeds up training on RTX cards!
    scaler = torch.amp.GradScaler('cuda')
    
    # --- Logging ---
    writer = SummaryWriter('runs/carla_multimodal_ai')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            vision = batch['vision'].to(DEVICE, non_blocking=True)
            bev = batch['bev'].to(DEVICE, non_blocking=True)
            meta = batch['meta'].to(DEVICE, non_blocking=True)
            targets = batch['actions'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda'):
                pred_actions, pred_aux = model(vision, bev, meta)
                
                # Main Action Loss
                action_loss = weighted_mse_loss(pred_actions, targets)
                
                # Auxiliary Loss: The true Speed is meta[:, 0], True Vehicle_Dist is meta[:, 6]
                target_aux = torch.stack([meta[:, 0], meta[:, 6]], dim=1)
                aux_loss = aux_criterion(pred_aux, target_aux)
                
                # Total Loss (Auxiliary is weighted at 50% so it doesn't overpower driving)
                loss = action_loss + 0.5 * aux_loss
                
            # Backward pass with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar_val:
                vision = batch['vision'].to(DEVICE, non_blocking=True)
                bev = batch['bev'].to(DEVICE, non_blocking=True)
                meta = batch['meta'].to(DEVICE, non_blocking=True)
                targets = batch['actions'].to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    pred_actions, pred_aux = model(vision, bev, meta)
                    action_loss = weighted_mse_loss(pred_actions, targets)
                    target_aux = torch.stack([meta[:, 0], meta[:, 6]], dim=1)
                    aux_loss = aux_criterion(pred_aux, target_aux)
                    loss = action_loss + 0.5 * aux_loss
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Tensorboard Logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> New best model found! Saving to best_model.pth")
            torch.save(model.state_dict(), 'best_model.pth')
            
    writer.close()
    print("Training Finished Successfully!")

if __name__ == '__main__':
    train()
