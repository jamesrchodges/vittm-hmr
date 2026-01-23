import os
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import TCGA_HMR_Dataset
from model import ViTTM_HMR

# --- CONFIGURATION ---
BATCH_SIZE = 64        
LEARNING_RATE = 1e-4   
EPOCHS = 20            
MASK_RATIO = 0.75      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MANIFEST_FILE = "full_tcga_manifest.csv"  
SLIDE_DIR = "/scratch/jh1466/TCGA_BRCA_Slides" 

def train():
    # Optimization: Use CuDNN Benchmark for speed
    torch.backends.cudnn.benchmark = True
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"{timestamp}_FULL_RUN"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"--- STARTING FULL RUN (3,000 Slides) ---")
    print(f"--- Architecture: ViTTM-HMR ---")
    print(f"--- Device: {DEVICE} | Batch: {BATCH_SIZE} ---")
    
    print("Loading Full Manifest...")
    try:
        dataset = TCGA_HMR_Dataset(
            manifest_file=MANIFEST_FILE, 
            slide_dir=SLIDE_DIR, 
            transform=None, 
            use_dummy_data=False
        )
        print(f"Total Training Patches: {len(dataset)}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find {MANIFEST_FILE}.")
        print("Please run the 'create_full_dataset_parallel.py' script first.")
        return
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=2,
        timeout=300
    )
    
    # 2. Initialize Model
    model = ViTTM_HMR().to(DEVICE)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=0.05)
    
    # Cosine Scheduler for smooth convergence over long runs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    loss_history = []
    
    # 3. Production Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = datetime.now()
        
        print(f"--- Epoch {epoch+1}/{EPOCHS} Started at {start_time.strftime('%H:%M:%S')} ---")
        
        for i, batch in enumerate(dataloader):
            # Load Data (Non-blocking for speed)
            memory_view = batch['memory_view'].to(DEVICE, non_blocking=True)
            process_view = batch['process_view'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward Pass 
            loss, pred_features, target_features = model(process_view, memory_view, mask_ratio=MASK_RATIO)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()

            # Saves progress every ~45 minutes (approx 5000 steps)
            if i > 0 and i % 5000 == 0:
                print(f"Saving safety checkpoint at step {i}...")
                safety_path = os.path.join(results_dir, "checkpoint_latest_safety.pth")
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, safety_path)
            
            # Log every 100 steps 
            if i % 100 == 0:
                display_loss = 1.0 + loss.item()
                print(f"Ep {epoch+1} | Step {i}/{len(dataloader)} | Loss: {loss.item():.4f} (Dist: {display_loss:.4f})")
                loss_history.append({'epoch': epoch+1, 'step': i, 'loss': loss.item()})

        # End of Epoch Stats
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        duration = datetime.now() - start_time
        
        print(f"=== EPOCH {epoch+1} COMPLETE | Avg Loss: {avg_loss:.4f} | Time: {duration} ===")
        
        # Save Checkpoint Every Epoch 
        save_path = os.path.join(results_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Checkpoint saved: {save_path}")
        
        # Save CSV log incrementally
        pd.DataFrame(loss_history).to_csv(os.path.join(results_dir, "training_log.csv"), index=False)

    print("FULL TRAINING RUN COMPLETE.")

if __name__ == "__main__":
    train()