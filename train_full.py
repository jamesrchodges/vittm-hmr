import os
import torch
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
import time

# Custom Modules
from dataloader import TCGA_HMR_Dataset
from model import ViTTM_HMR

# --- CONFIGURATION ---
MANIFEST_FILE = "full_tcga_manifest.csv" # The new large file
SLIDE_DIR = "/scratch/jh1466/TCGA_BRCA_Slides"
RESULTS_DIR = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_FULL_RUN"

BATCH_SIZE = 64        
LEARNING_RATE = 1.5e-4 
EPOCHS = 20            
MASK_RATIO = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 16     

def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"--- STARTING FULL TRAINING RUN ---")
    print(f"Output Directory: {RESULTS_DIR}")
    
    # 1. Dataset
    print("Loading Manifest...")
    dataset = TCGA_HMR_Dataset(
        manifest_file=MANIFEST_FILE, 
        slide_dir=SLIDE_DIR, 
        transform=None
    )
    print(f"Total Patches: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    
    # 2. Model
    model = ViTTM_HMR().to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=0.05)
    
    # Scheduler: Linear warmup for 1 epoch, then Cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(dataloader), epochs=EPOCHS
    )
    
    # Mixed Precision Scaler
    scaler = GradScaler()

    # 3. Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            process_view = batch['process_view'].to(DEVICE)
            memory_view = batch['memory_view'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Mixed Precision Context
            with autocast():
                loss, _, _ = model(process_view, memory_view, mask_ratio=MASK_RATIO)
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # Logging (Every 100 steps)
            if i % 100 == 0:
                print(f"Ep {epoch+1} | Step {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
                # Append to log file immediately (safe against crashes)
                with open(os.path.join(RESULTS_DIR, "train_log.csv"), "a") as f:
                    f.write(f"{epoch+1},{i},{loss.item()}\n")

        # End of Epoch Stats
        duration = (time.time() - start_time) / 60
        avg_loss = epoch_loss / len(dataloader)
        print(f"=== EPOCH {epoch+1} COMPLETE | Avg Loss: {avg_loss:.4f} | Time: {duration:.1f} min ===")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    # Initialize Log File
    with open(os.path.join(RESULTS_DIR, "train_log.csv"), "w") as f:
        f.write("Epoch,Step,Loss\n")
        
    train()