import os
import torch
import torch.optim as optim
import pandas as pd
import webdataset as wds
import numpy as np
from datetime import datetime
from torchvision import transforms
from model import ViTTM_HMR

# --- CONFIGURATION ---
BATCH_SIZE = 256        
LEARNING_RATE = 1e-4   
EPOCHS = 20            
MASK_RATIO = 0.75      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_URL = "/scratch/jh1466/wds_data/shard-worker*-*.tar" 

NUM_BATCHES_PER_EPOCH = 192000 # Approx 1 epoch

import torch

class TorchVahadaneNormalizer(torch.nn.Module):
    def __init__(self, target_path="reference_patch.png", device='cuda'):
        super().__init__()
        self.device = device
        
        # Load reference image once and calculate its statistics
        # Target: roughly Pink/Purple separation
        self.target_fit = True
        
        # Pre-calculated values from a standard TCGA good patch
   
        self.target_HE = torch.tensor([
            [0.5626, 0.2159], 
            [0.7201, 0.8012], 
            [0.4062, 0.5581]
        ], device=device)
        
        self.max_C_target = torch.tensor([1.9705, 1.0308], device=device)

    def get_stain_matrix(self, I, beta=0.15):
        # I: [N, 3] (Flattened pixels)
        # Convert to Optical Density (OD)
        OD = -torch.log(I + 1e-6)
        
        # Remove transparent background (OD < beta)
        mask = (OD > beta).all(dim=1)
        OD_hat = OD[mask]
        
        if OD_hat.shape[0] < 10: 
            return None
        
        # Eigen decomposition of covariance matrix
        cov = torch.cov(OD_hat.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        
        # Extract two largest eigenvectors (Stain vectors)
        eigvecs = eigvecs[:, [2, 1]] # Top 2
        
        # Ensure positive direction
        if eigvecs[0, 0] < 0: eigvecs[:, 0] *= -1
        if eigvecs[0, 1] < 0: eigvecs[:, 1] *= -1
        
        # Project
        T_hat = torch.mm(OD_hat, eigvecs)
        phi = torch.atan2(T_hat[:, 1], T_hat[:, 0])
        
        min_phi = torch.quantile(phi, 0.01)
        max_phi = torch.quantile(phi, 0.99)
        
        v_min = torch.matmul(eigvecs, torch.stack((torch.cos(min_phi), torch.sin(min_phi))))
        v_max = torch.matmul(eigvecs, torch.stack((torch.cos(max_phi), torch.sin(max_phi))))
        
        # Heuristic to order H (Hematoxylin) vs E (Eosin)
        # H usually has higher first component
        if v_min[0] > v_max[0]:
            HE = torch.stack((v_min, v_max), dim=1)
        else:
            HE = torch.stack((v_max, v_min), dim=1)
            
        return HE

    def forward(self, images):
        # images: [B, 3, H, W] - Normalized (0-1) Tensors
        B, C, H, W = images.shape
        output = []
        
        for i in range(B):
            img_flat = images[i].permute(1, 2, 0).reshape(-1, 3) # [HxW, 3]
            img_flat = torch.clamp(img_flat, 1e-6, 1.0)
            
            # 1. Get Stain Matrix
            HE = self.get_stain_matrix(img_flat)
            
            if HE is None: 
                # Background / Failed detection -> Keep original
                output.append(images[i])
                continue
            
            # 2. Get Concentration
            OD = -torch.log(img_flat)
            # Least squares to find concentrations
            # C = (HE^T . HE)^-1 . HE^T . OD
            solution = torch.linalg.lstsq(HE, OD.T).solution # [2, N]
            C = solution.T
            
            # 3. Normalize Concentration
            max_C = torch.quantile(C, 0.99, dim=0)
            C_norm = C * (self.max_C_target / (max_C + 1e-6))
            
            # 4. Reconstruct
            I_norm_OD = torch.mm(self.target_HE, C_norm.T).T
            I_norm = torch.exp(-I_norm_OD)
            I_norm = torch.clamp(I_norm, 0, 1)
            
            output.append(I_norm.reshape(H, W, 3).permute(2, 0, 1))
            
        return torch.stack(output)

# --- TRANSFORMS ---
class ECT_Transform:
    """Adapted ECT Transform for WebDataset"""
    def __init__(self, target_size=256, shift_limit=0.1):
        self.target_size = target_size
        self.shift_limit = shift_limit 

    def __call__(self, img):
        w, h = img.size
        max_dx = int(w * self.shift_limit)
        max_dy = int(h * self.shift_limit)
        dx = np.random.randint(-max_dx, max_dx + 1)
        dy = np.random.randint(-max_dy, max_dy + 1)
        center_x, center_y = w // 2 + dx, h // 2 + dy
        left = center_x - self.target_size // 2
        top = center_y - self.target_size // 2
        return img.crop((left, top, left + self.target_size, top + self.target_size))

def process_sample(sample):
    """
    Takes a sample (dict), applies transforms, returns the two views.
    """
    image = sample["jpg"] # This is a PIL image
    
    # 1. Apply ECT (Random Crop/Shift)
    ect = ECT_Transform(target_size=256)
    img_view = ect(image)
    
    # 2. Standard Augmentations (Flip/Rotate) 
    if np.random.rand() > 0.5: img_view = img_view.transpose(0) # Flip L/R
    
    # 3. Normalize & Tensor
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    memory_input = normalize(img_view)
    process_input = memory_input.clone()
    
    return memory_input, process_input

def make_loader():
    # The WebDataset Pipeline
    dataset = (
        wds.WebDataset(DATA_URL, nodesplitter=wds.split_by_node, shardshuffle=True)
        .shuffle(5000)        # Shuffle samples in buffer
        .decode("pil")        # Decode JPG to PIL
        .map(process_sample)  # Apply ECT and formatting
        .to_tuple()           # Convert to tuple (mem, proc)
    )
    
    loader = wds.WebLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=8,        
        pin_memory=True
    )
    
    return loader

def train():
    torch.backends.cudnn.benchmark = True
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"{timestamp}_WDS_RUN"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"--- STARTING FULL RUN (WebDataset Mode) ---")
    print(f"--- Device: {DEVICE} | Batch: {BATCH_SIZE} ---")
    
    # Initialize Model
    model = ViTTM_HMR().to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Initialize Loader
    print("Initializing WebDataset Loader...")
    dataloader = make_loader()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = datetime.now()
        
        print(f"--- Epoch {epoch+1}/{EPOCHS} Started ---")
                
        step_count = 0
        # Initialize Normaliser on GPU
        normalizer = TorchVahadaneNormalizer(device=DEVICE)

        for i, (memory_view, process_view) in enumerate(dataloader):
            # 1. Move raw data to GPU
            memory_view = memory_view.to(DEVICE, non_blocking=True)
            process_view = process_view.to(DEVICE, non_blocking=True)
            
            # Note: Only normalise the MEMORY view (High Res). 
            with torch.no_grad():
                memory_view = normalizer(memory_view)
                # Optional: Normalize process_view too if desired
                process_view = normalizer(process_view)
            
            # 3. Standard Forward Pass
            optimizer.zero_grad()
            loss, _, _ = model(process_view, memory_view, mask_ratio=MASK_RATIO)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1
            
            if i % 100 == 0:
                print(f"Ep {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
                loss_history.append({'epoch': epoch+1, 'step': i, 'loss': loss.item()})

            # Safety Checkpoint
            if i > 0 and i % 5000 == 0:
                print(f"Saving safety checkpoint...")
                torch.save(model.state_dict(), os.path.join(results_dir, "ckpt_latest.pth"))
        
        # End of Epoch
        scheduler.step()
        duration = datetime.now() - start_time
        avg_loss = epoch_loss / step_count if step_count > 0 else 0
        print(f"=== EPOCH {epoch+1} DONE | Avg Loss: {avg_loss:.4f} | Time: {duration} ===")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
        }, os.path.join(results_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        pd.DataFrame(loss_history).to_csv(os.path.join(results_dir, "training_log.csv"), index=False)

if __name__ == "__main__":
    train()