import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import openslide

class ECT_Transform:
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

class TCGA_HMR_Dataset(Dataset):
    def __init__(self, manifest_file, slide_dir, patch_size=256, transform=None, use_dummy_data=False):
        self.slide_dir = slide_dir
        self.patch_size = patch_size
        self.transform = transform
        self.use_dummy_data = use_dummy_data
        
        if not self.use_dummy_data and os.path.exists(manifest_file):
            self.coords = pd.read_csv(manifest_file)
        else:
            print("Using DUMMY data for testing")
            n = 10
            self.coords = pd.DataFrame({
                'slide_id': ['dummy_slide'] * n,
                'x': [0] * n,
                'y': [0] * n
            })
        self.slide_cache = {}

    def _get_slide_handle(self, slide_path):
        if slide_path not in self.slide_cache:
            self.slide_cache[slide_path] = openslide.OpenSlide(slide_path)
        return self.slide_cache[slide_path]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        slide_id = row['slide_id']
        
        # 1. LOAD RAW DATA
        if self.use_dummy_data:
            patch = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        else:
            slide_path = os.path.join(self.slide_dir, f"{slide_id}.svs")
            slide = self._get_slide_handle(slide_path)
            read_size = int(self.patch_size * 1.2) 
            patch = slide.read_region(
                (int(row['x']), int(row['y'])), 
                0, 
                (read_size, read_size)
            ).convert('RGB')

        # 2. APPLY DOMAIN ENHANCEMENTS (ECT)
        ect = ECT_Transform(target_size=self.patch_size)
        img_view = ect(patch) # Returns PIL Image
        
        if self.transform:
            img_view = self.transform(img_view)

        # 3. PREPARE STREAMS (Normalization & Resizing)
        to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Stream A: High-Res Memory View
        memory_input = to_tensor_norm(img_view)

        # Stream B: Low-Res Process View
        process_input = memory_input.clone()
        
        return {
            'memory_view': memory_input,  
            'process_view': process_input, 
            'slide_id': slide_id,
            'coords': (row['x'], row['y'])
        }