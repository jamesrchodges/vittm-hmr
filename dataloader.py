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
            print(f"Loading manifest: {manifest_file}")
            self.coords = pd.read_csv(manifest_file)
            self.has_filepath = 'file_path' in self.coords.columns
        else:
            print("Using DUMMY data for testing")
            n = 10
            self.coords = pd.DataFrame({
                'slide_id': ['dummy_slide'] * n,
                'file_path': ['dummy.svs'] * n,
                'x': [0] * n,
                'y': [0] * n
            })
            self.has_filepath = True
            
        self.slide_cache = {}
        # Max slides to keep open per worker to prevent OOM
        self.CACHE_SIZE = 50 

    def _get_slide_handle(self, slide_path):
        # 1. Cache Hit
        if slide_path in self.slide_cache:
            return self.slide_cache[slide_path]
            
        # 2. Cache Cleanup 
        if len(self.slide_cache) >= self.CACHE_SIZE:
            # Remove the oldest entry (FIFO)
            oldest_key = next(iter(self.slide_cache))
            try:
                self.slide_cache.pop(oldest_key).close()
            except:
                pass # Already closed or error
        
        # 3. Open New Slide
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide file missing: {slide_path}")

        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            raise RuntimeError(f"OpenSlide failed to open {slide_path}: {e}")
            
        self.slide_cache[slide_path] = slide
        return slide

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        slide_id = row['slide_id']
        
        try:
            # 1. LOAD RAW DATA
            if self.use_dummy_data:
                patch = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
            else:
                # PATH CONSTRUCTION
                if self.has_filepath:
                    rel_path = row['file_path']
                    slide_path = os.path.join(self.slide_dir, rel_path)
                else:
                    slide_path = os.path.join(self.slide_dir, f"{slide_id}.svs")

                slide = self._get_slide_handle(slide_path)
                read_size = int(self.patch_size * 1.2) 
                
                # READ REGION
                patch = slide.read_region(
                    (int(row['x']), int(row['y'])), 
                    0, 
                    (read_size, read_size)
                ).convert('RGB')

            # 2. APPLY DOMAIN ENHANCEMENTS (ECT)
            ect = ECT_Transform(target_size=self.patch_size)
            img_view = ect(patch) 
            
            if self.transform:
                img_view = self.transform(img_view)

            # 3. PREPARE STREAMS
            to_tensor_norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            memory_input = to_tensor_norm(img_view)
            process_input = memory_input.clone()
            
            return {
                'memory_view': memory_input,  
                'process_view': process_input, 
                'slide_id': slide_id,
                'coords': (row['x'], row['y'])
            }

        except Exception as e:
            print(f"Skipping bad patch in {slide_id}: {e}")
            dummy = torch.zeros((3, self.patch_size, self.patch_size))
            return {
                'memory_view': dummy,
                'process_view': dummy,
                'slide_id': "error",
                'coords': (0, 0)
            }