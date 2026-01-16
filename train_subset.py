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
EPOCHS = 10 
MASK_RATIO = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_FILE = "patches_subset_300.csv"
SLIDE_DIR = "/scratch/jh1466/TCGA_BRCA_Slides"

def train():
    torch.backends.cudnn.benchmark = True
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"{timestamp}_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"--- Starting ViTTM-HMR Training (FP32 Stable Mode) ---")
    print(f"--- Mask Ratio: {MASK_RATIO} | Device: {DEVICE} | Batch: {BATCH_SIZE} ---")
    
    dataset = TCGA_HMR_Dataset(
        manifest_file=MANIFEST_FILE, 
        slide_dir=SLIDE_DIR, 
        transform=None, 
        use_dummy_data=False
    )
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=2
    )
    
    model = ViTTM_HMR().to(DEVICE)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=0.05)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # scaler = GradScaler()

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(dataloader):
            memory_view = batch['memory_view'].to(DEVICE, non_blocking=True)
            process_view = batch['process_view'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            loss, pred_features, target_features = model(process_view, memory_view, mask_ratio=MASK_RATIO)
            
            # --- Backward Pass ---
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 50 == 0:
                display_loss = 1.0 + loss.item()
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(dataloader)}] Raw Loss: {loss.item():.4f} (Dist: {display_loss:.4f})")
                loss_history.append({'epoch': epoch+1, 'step': i, 'loss': loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Avg Loss: {avg_loss:.4f} ===")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(results_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

    df = pd.DataFrame(loss_history)
    df.to_csv(os.path.join(results_dir, "loss_log.csv"), index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['loss'], label='Cosine Sim Loss (Target: -1.0)')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("HMR Feature Distillation Training")
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    
    print("Training Complete.")

if __name__ == "__main__":
    train()