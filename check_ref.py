import os
import json
from tqdm import tqdm

# --- Configuration ---
DATASET_ROOT = "/BS/DApt/work/fashion-iq/fashionIQ_dataset"
FASHION_IQ_DATA_FILE = os.path.join(DATASET_ROOT, "captions", "cap.shirt.val.json")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")

# --- SETTINGS ---
# Set to an integer (e.g., 500) to limit the gallery, 
# or set to None to process the entire dataset.
TARGET_SET_SIZE = None  #500

OUTPUT_GALLERY_FILE = "/BS/DApt/work/fashion-iq/closed_set_image_ids.json"
OUTPUT_QUERIES_FILE = "/BS/DApt/work/fashion-iq/closed_set_queries.json"

print(f"Starting closed-set creation process (Limit: {TARGET_SET_SIZE})...")

# --- Step 1: Scan for files ---
print(f"Scanning for existing images in: {IMAGES_DIR}")
existing_images_no_ext = {os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR)}

# --- Step 2: Load Data ---
with open(FASHION_IQ_DATA_FILE, 'r') as f:
    fashion_iq_data = json.load(f)

# --- Step 3: Build the gallery ---
valid_triplets = []
for triplet in fashion_iq_data:
    if triplet.get('candidate') in existing_images_no_ext and \
       triplet.get('target') in existing_images_no_ext:
        valid_triplets.append(triplet)

selected_image_ids = set()
for triplet in tqdm(valid_triplets, desc="Selecting images"):
    selected_image_ids.add(triplet['candidate'])
    selected_image_ids.add(triplet['target'])
    
    # Logic update: Only break if TARGET_SET_SIZE is not None
    if TARGET_SET_SIZE is not None and len(selected_image_ids) >= TARGET_SET_SIZE:
        break

final_gallery_list = sorted(list(selected_image_ids))
final_gallery_set = set(final_gallery_list)

print(f"Gallery size: {len(final_gallery_list)} images.")

# --- Step 4: Filter Queries ---
selected_queries = []
for triplet in tqdm(valid_triplets, desc="Processing queries"):
    # Only keep queries where both images made it into the gallery
    if triplet['candidate'] in final_gallery_set and triplet['target'] in final_gallery_set:
        for mod_text in triplet['captions']:
            selected_queries.append({
                "reference_id": triplet['candidate'],
                "modification_text": mod_text,
                "target_id": triplet['target']
            })

# --- Step 5: Save ---
with open(OUTPUT_GALLERY_FILE, 'w') as f:
    json.dump(final_gallery_list, f, indent=2)

with open(OUTPUT_QUERIES_FILE, 'w') as f:
    json.dump(selected_queries, f, indent=2)

print(f"\n✅ Done. Saved {len(final_gallery_list)} images and {len(selected_queries)} valid queries.")