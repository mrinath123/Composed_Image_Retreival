import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# --- 1. Configuration ---
DATASET_ROOT = "/BS/DApt/work/fashion-iq/fashionIQ_dataset"
MODEL_ID = "google/gemma-3-4b-it"
OUTPUT_FILE = "/BS/DApt/work/fashion-iq/base_captions_v2.json"
VERIFIED_IDS_FILE = "/BS/DApt/work/fashion-iq/closed_set_image_ids.json"

# --- 2. Load Model and Processor ---
print(f"Loading model: {MODEL_ID}")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. Load Verified Image IDs ---
print(f"Loading verified image IDs from: {VERIFIED_IDS_FILE}")
try:
    with open(VERIFIED_IDS_FILE, 'r') as f:
        sample_image_ids = json.load(f)
    print(f"Loaded {len(sample_image_ids)} verified image IDs to process.")
except FileNotFoundError:
    print(f"Error: Verified IDs file not found at {VERIFIED_IDS_FILE}. Run create_verified_ids.py first.")
    exit()

# --- 4. Generate Captions ---
images_dir = os.path.join(DATASET_ROOT, "images")
results = []


for image_id in tqdm(sample_image_ids, desc="Generating Captions"):
    image_path = os.path.join(images_dir, f"{image_id}.jpg")
    
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Verified image file not found: {image_path}. Skipping.")
        continue

    #
    # v v v v v v v v v v v v v v v v v v v v v v v v v v v
    # THIS IS THE CORRECTED PART
    # v v v v v v v v v v v v v v v v v v v v v v v v v v v
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Pass the actual PIL image object here
                {"type": "text", "text": "Describe only the clothing item in this image. Ignore the person wearing it. Focus on the garment's color, style, and any visible text or graphics. In a single, concise sentence."}
            ]
        }
    ]
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
    # END OF CORRECTION
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
    
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        caption = processor.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
        
        results.append({
            "image_id": image_id,
            "caption": caption
        })
        
    except Exception as e:
        print(f"\nError during model generation for image {image_id}: {e}")
        continue

# --- 5. Save Results to JSON ---
if results:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Success! Saved {len(results)} base captions to: {OUTPUT_FILE}")
else:
    print("\n❌ No captions were generated.")