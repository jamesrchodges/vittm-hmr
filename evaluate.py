import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score
import h5py
from PIL import Image
import numpy as np

from model import ViTTM_HMR

# --- CONFIGURATION ---
BATCH_SIZE = 128  
LR_EVAL = 1e-3
EPOCHS_EVAL = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/scratch/jh1466/camelyon/" 
CHECKPOINT_PATH = "20260114_131710_results/checkpoint_epoch_10.pth" 

class LinearProbe(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Backbone input: 256x256 -> Output: 384-dim features
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        # --- OPTION B INFERENCE LOGIC ---
        
        # 1. Embed Process Tokens (Student)
        proc_tokens = self.backbone.process_embed(x)
        proc_tokens = proc_tokens.flatten(2).transpose(1, 2)
        proc_tokens = proc_tokens + self.backbone.process_pos_embed
        
        # 2. Initialize Latent Memory
        B = proc_tokens.shape[0]
        curr_memory = self.backbone.memory_bank.expand(B, -1, -1) 
        
        # 3. ViTTM Loop (Read -> Compute -> Write)
        for blk in self.backbone.blocks:
            proc_tokens, curr_memory = blk(x_process=proc_tokens, x_memory=curr_memory)
            
        # 4. Norm
        features = self.backbone.norm(proc_tokens) # [B, 16, 384]
        
        # 5. Global Average Pooling -> Classification
        global_feat = features.mean(dim=1) # [B, 384]
        return self.head(global_feat)

class LocalPCamDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, limit=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.limit = limit
        
        self.x_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{split}_x.h5')
        self.y_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{split}_y.h5')
        
        if not os.path.exists(self.x_path) or not os.path.exists(self.y_path):
            raise FileNotFoundError(f"Could not find HDF5 files in {root_dir}.")

        with h5py.File(self.x_path, 'r') as f:
            self.total_len = f['x'].shape[0]
            
        self.x_file = None
        self.y_file = None

    def __len__(self):
        if self.limit:
            return min(self.limit, self.total_len)
        return self.total_len

    def __getitem__(self, idx):
        if self.x_file is None:
            self.x_file = h5py.File(self.x_path, 'r')
            self.y_file = h5py.File(self.y_path, 'r')
            
        img_arr = self.x_file['x'][idx]
        label_arr = self.y_file['y'][idx]
        
        image = Image.fromarray(img_arr)
        label = int(label_arr.flatten()[0])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def evaluate_pcam():
    torch.backends.cudnn.benchmark = True
    
    print(f"--- Starting Linear Probe (FP32 Safe Mode) ---")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    try:
        print("Initializing Train Dataset...")
        train_set = LocalPCamDataset(root_dir=DATA_DIR, split='train', transform=transform, limit=5000)
        print("Initializing Test Dataset...")
        test_set = LocalPCamDataset(root_dir=DATA_DIR, split='test', transform=transform, limit=1000)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        return

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 2. Load Model
    print(f"Loading backbone from: {CHECKPOINT_PATH}")
    backbone = ViTTM_HMR().to(DEVICE)
    
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        backbone.load_state_dict(state_dict)
    except Exception as e:
        print(f"Loading state dict error: {e}")
        print("Attempting strict=False...")
        backbone.load_state_dict(state_dict, strict=False)
    
    model = LinearProbe(backbone).to(DEVICE)
    optimizer = optim.Adam(model.head.parameters(), lr=LR_EVAL)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Train Probe
    print("Training Linear Head...")
    for epoch in range(EPOCHS_EVAL):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            # Load data
            images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward Pass 
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward Pass 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")

    # 4. Final Eval
    print("\nRunning Final Evaluation on Test Set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"--- Final PCam Accuracy: {acc*100:.2f}% ---")
    
    if acc > 0.65:
        print("SUCCESS: The Latent Memory features are valid!")
    elif acc > 0.55:
        print("WEAK SIGNAL")
    else:
        print("FAILURE")

if __name__ == "__main__":
    evaluate_pcam()