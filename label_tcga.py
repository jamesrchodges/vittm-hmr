import pandas as pd
import os

df = pd.read_csv("patches_subset_300.csv")

def get_label(slide_id):
    try:
        # 1. CLEAN THE ID: Get only the filename after the last '/'
        filename = slide_id.split('/')[-1] 
        
        # 2. PARSE TCGA BARCODE
        # Split: ['TCGA', 'C8', 'A12O', '01Z', '00', 'DX1'...]
        parts = filename.split('-')
        
        # Take the first 2 chars and convert to int
        sample_code = int(parts[3][:2]) 
        
        if 1 <= sample_code <= 9:
            return 1 # TUMOR 
        elif 10 <= sample_code <= 19:
            return 0 # NORMAL 
        else:
            return -1 # Other/Control
    except Exception as e:
        print(f"Error parsing {slide_id}: {e}")
        return -1

df['label'] = df['slide_id'].apply(get_label)

# Filter out failures
df = df[df['label'] != -1]

# Save
df.to_csv("patches_subset_300_labeled.csv", index=False)
print("Labeled CSV created")
print(df['label'].value_counts())