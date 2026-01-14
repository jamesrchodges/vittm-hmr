import os
import glob
import pandas as pd
import numpy as np
import openslide
import cv2
from multiprocessing import Pool, cpu_count
import time

# --- CONFIGURATION ---
SLIDE_DIR = "/scratch/jh1466/TCGA_BRCA_Slides" 
OUTPUT_CSV = "full_tcga_manifest.csv"
PATCH_SIZE = 256
TISSUE_THRESHOLD = 0.5
THUMBNAIL_LEVEL = 32
NUM_WORKERS = min(32, cpu_count()) 

def get_tissue_mask(slide_path):
    """
    Returns valid coordinates for a SINGLE slide.
    Global function required for multiprocessing pickling.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        w, h = slide.dimensions
        
        # 1. Thumbnail Check
        # If slide is too small, skip
        if w < 2000 or h < 2000:
            return []

        thumb_w, thumb_h = w // THUMBNAIL_LEVEL, h // THUMBNAIL_LEVEL
        if thumb_w < 1 or thumb_h < 1: return []

        thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB')
        thumb_arr = np.array(thumbnail)

        # 2. Otsu Thresholding (Saturation)
        hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        _, tissue_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Extract Coords
        valid_coords = []
        mask_h, mask_w = tissue_mask.shape
        step = PATCH_SIZE // THUMBNAIL_LEVEL
        
        rel_path = os.path.relpath(slide_path, SLIDE_DIR)
        slide_id = os.path.splitext(os.path.basename(slide_path))[0]

        for y in range(0, mask_h - step, step):
            for x in range(0, mask_w - step, step):
                # Check tissue percentage in this patch
                region = tissue_mask[y : y+step, x : x+step]
                if np.sum(region) / (step**2 * 255) >= TISSUE_THRESHOLD:
                    valid_coords.append({
                        'slide_id': slide_id,
                        'file_path': rel_path, # Store partial path to find file later
                        'x': int(x * THUMBNAIL_LEVEL),
                        'y': int(y * THUMBNAIL_LEVEL)
                    })
        
        slide.close()
        return valid_coords

    except Exception as e:
        print(f"Error processing {os.path.basename(slide_path)}: {e}")
        return []

def main():
    start_time = time.time()
    print(f"--- Starting Parallel Tiling (Workers: {NUM_WORKERS}) ---")
    
    # 1. Find all slides
    print(f"Scanning {SLIDE_DIR}...")
    all_svs = glob.glob(os.path.join(SLIDE_DIR, "**/*.svs"), recursive=True)
    print(f"Found {len(all_svs)} slides.")
    
    # 2. Parallel Process
    all_patches = []
    processed_count = 0
    
    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap_unordered(get_tissue_mask, all_svs, chunksize=1):
            if result:
                all_patches.extend(result)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(all_svs)} slides... (Total Patches: {len(all_patches)})")

    # 3. Save
    print("Saving to CSV...")
    df = pd.DataFrame(all_patches)
    df.to_csv(OUTPUT_CSV, index=False)
    
    duration = (time.time() - start_time) / 3600
    print(f"\nDONE! Created {OUTPUT_CSV} with {len(df)} patches.")
    print(f"Total time: {duration:.2f} hours.")

if __name__ == "__main__":
    main()