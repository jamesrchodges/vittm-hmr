import os
import glob
import openslide

DOWNLOAD_DIR = "/scratch/jh1466/TCGA_BRCA_Slides" 

# 2. Find one .svs file 
print(f"Searching for .svs files in {DOWNLOAD_DIR}...")
files = glob.glob(os.path.join(DOWNLOAD_DIR, "**/*.svs"), recursive=True)

if not files:
    print("ERROR: No .svs files found! Check your path.")
else:
    print(f"Found {len(files)} slides. Testing the first one...")
    test_slide_path = files[0]
    print(f"Attempting to open: {test_slide_path}")

    try:
        # 3. Open the slide
        slide = openslide.OpenSlide(test_slide_path)
        
        # 4. Print Metadata 
        w, h = slide.dimensions
        print(f"\n--- SUCCESS! Data is readable ---")
        print(f"Dimensions: {w} x {h} pixels")
        print(f"Level Count: {slide.level_count}")
        print(f"Objective Power: {slide.properties.get('openslide.objective-power', 'Unknown')}x")
        
        # 5. Check for 'Biospecimen' vs 'Clinical' error
        # If dimensions are small (e.g., 1000x1000), you might have downloaded snapshots, not slides.
        if w < 10000 or h < 10000:
            print("\nWARNING: Image seems too small for a WSI")
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not read slide.\n{e}")
        