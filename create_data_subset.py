import os
import glob
import pandas as pd
import numpy as np
import openslide
import cv2
import random

# --- CONFIGURATION ---
SLIDE_DIR = "/scratch/jh1466/TCGA_BRCA_Slides"  
OUTPUT_CSV = "patches_subset_300.csv"           
SAMPLE_SIZE = 300                               
PATCH_SIZE = 256                        
TISSUE_THRESHOLD = 0.5                  
THUMBNAIL_LEVEL_DOWNSAMPLE = 32         

def get_tissue_mask(slide, downsample_factor=32):
    """
    Generates a binary mask where 1=Tissue, 0=Background.
    """
    w, h = slide.dimensions
    thumb_w, thumb_h = w // downsample_factor, h // downsample_factor
    
    # Handle tiny slide edge cases
    if thumb_w < 1 or thumb_h < 1: return np.zeros((1,1))
    
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB')
    thumb_arr = np.array(thumbnail)
    
    # Simple Otsu thresholding on Saturation channel
    hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    _, tissue_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Cleanup noise
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    
    return tissue_mask / 255.0

def process_slide(slide_path):
    """
    Scans a single slide and returns a list of valid (x, y) coordinates.
    """
    slide = openslide.OpenSlide(slide_path)
    
    # Use Relative Path as ID so DataLoader can find nested files
    rel_path = os.path.relpath(slide_path, SLIDE_DIR)
    slide_id = os.path.splitext(rel_path)[0]
    
    mask = get_tissue_mask(slide, downsample_factor=THUMBNAIL_LEVEL_DOWNSAMPLE)
    mask_h, mask_w = mask.shape
    valid_coords = []
    
    step_size_on_mask = PATCH_SIZE // THUMBNAIL_LEVEL_DOWNSAMPLE
    
    for y_mask in range(0, mask_h - step_size_on_mask, step_size_on_mask):
        for x_mask in range(0, mask_w - step_size_on_mask, step_size_on_mask):
            patch_region = mask[y_mask : y_mask + step_size_on_mask, 
                                x_mask : x_mask + step_size_on_mask]
            
            if np.sum(patch_region) / (step_size_on_mask ** 2) >= TISSUE_THRESHOLD:
                valid_coords.append({
                    'slide_id': slide_id,
                    'x': int(x_mask * THUMBNAIL_LEVEL_DOWNSAMPLE),
                    'y': int(y_mask * THUMBNAIL_LEVEL_DOWNSAMPLE)
                })
                
    return valid_coords

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"1. Scanning for slides in {SLIDE_DIR}...")
    # Recursive search to find nested .svs files
    all_svs_files = glob.glob(os.path.join(SLIDE_DIR, "**/*.svs"), recursive=True)
    total_files = len(all_svs_files)
    print(f"   -> Found {total_files} total slides.")
    
    # --- SUBSET LOGIC ---
    if total_files > SAMPLE_SIZE:
        print(f"2. Randomly selecting {SAMPLE_SIZE} representative slides...")
        random.seed(42) 
        selected_files = random.sample(all_svs_files, SAMPLE_SIZE)
    else:
        print(f"2. Dataset is smaller than {SAMPLE_SIZE}. Using all slides.")
        selected_files = all_svs_files

    # --- TILING LOOP ---
    print(f"3. Processing subset ({len(selected_files)} slides)...")
    all_patches = []
    
    for i, slide_path in enumerate(selected_files):
        print(f"   [{i+1}/{len(selected_files)}] Tiling: {os.path.basename(slide_path)}...")
        try:
            coords = process_slide(slide_path)
            all_patches.extend(coords)
        except Exception as e:
            print(f"   -> ERROR processing {os.path.basename(slide_path)}: {e}")

    # --- SAVE ---
    df = pd.DataFrame(all_patches)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSUCCESS. Subset dataset created: {OUTPUT_CSV}")
    print(f"Total Patches: {len(df)}")
