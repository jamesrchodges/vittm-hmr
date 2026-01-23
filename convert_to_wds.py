import os
import pandas as pd
import numpy as np
import openslide
import webdataset as wds
from multiprocessing import Process
import time
from PIL import Image
import io

# --- CONFIGURATION ---
MANIFEST_FILE = "full_tcga_manifest.csv"
SLIDE_DIR = "/scratch/jh1466/TCGA_BRCA_Slides"
OUTPUT_DIR = "/scratch/jh1466/wds_data"
NUM_WORKERS = 16  
PATCH_SIZE = 256
SAMPLES_PER_SHARD = 5000  # Standard WebDataset shard size

def write_partition(rank, df_partition, output_dir):
    """
    Worker function: Writes a subset of slides to .tar shards.
    """
    # Create a pattern unique to this worker
    pattern = os.path.join(output_dir, f"shard-worker{rank}-%05d.tar")
    
    # Initialize ShardWriter
    # maxcount: Start a new .tar file after 5000 images
    # maxsize: Start a new .tar file after 3GB (safety limit)
    sink = wds.ShardWriter(pattern, maxcount=SAMPLES_PER_SHARD, maxsize=3e9)
    
    # Group patches by slide as to only open each .svs file ONCE
    grouped = df_partition.groupby('slide_id')
    
    total_processed = 0
    
    for slide_id, group in grouped:
        try:
            # 1. Resolve Path
            if 'file_path' in group.columns:
                rel_path = group.iloc[0]['file_path']
                slide_path = os.path.join(SLIDE_DIR, rel_path)
            else:
                slide_path = os.path.join(SLIDE_DIR, f"{slide_id}.svs")
            
            if not os.path.exists(slide_path):
                print(f"[Worker {rank}] MISSING: {slide_path}")
                continue
                
            # 2. Open Slide Once
            slide = openslide.OpenSlide(slide_path)
            
            # 3. Extract All Patches for this Slide
            for _, row in group.iterrows():
                x, y = int(row['x']), int(row['y'])
                
                # Read Region (slightly larger to avoid edge artifacts, or exact size)
                try:
                    patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                except Exception:
                    continue # Skip bad regions

                # 4. Write to Shard
                # Key must be unique: slide_id + coords
                key = f"{slide_id}_{x}_{y}"
                
                sample = {
                    "__key__": key,
                    "jpg": patch,  # Automatically compresses
                    "json": {"slide_id": slide_id, "x": x, "y": y} # Metadata
                }
                sink.write(sample)
                total_processed += 1
                
            slide.close()
            
            if total_processed % 1000 == 0:
                print(f"[Worker {rank}] Processed {total_processed} patches...")

        except Exception as e:
            print(f"[Worker {rank}] ERROR on {slide_id}: {e}")

    sink.close()
    print(f"[Worker {rank}] DONE. Total: {total_processed}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading manifest: {MANIFEST_FILE}")
    df = pd.read_csv(MANIFEST_FILE)
    total_patches = len(df)
    print(f"Total Patches to Convert: {total_patches}")
    
    # Shuffle slides randomly to ensure shards are well-mixed
    unique_slides = df['slide_id'].unique()
    np.random.shuffle(unique_slides)
    
    # Split slides among workers
    slide_chunks = np.array_split(unique_slides, NUM_WORKERS)
    
    processes = []
    
    print(f"Starting {NUM_WORKERS} workers...")
    for i in range(NUM_WORKERS):
        # Filter DF for this worker's slides
        slides_for_worker = slide_chunks[i]
        df_partition = df[df['slide_id'].isin(slides_for_worker)]
        
        p = Process(target=write_partition, args=(i, df_partition, OUTPUT_DIR))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("CONVERSION COMPLETE.")

if __name__ == "__main__":
    main()