import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval

# --- 1. SETUP ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
DEVICE = "cuda"
BATCH_SIZE = 32 

IMAGES_DIR = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/images"
CAPTION_DIR = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/captions"
# Path to YOUR generated captions file
SYNTHETIC_CAPTIONS_FILE = "/BS/DApt/work/fashion-iq/modified_captions_short_v2.json"

CATEGORIES = ["dress", "toptee", "shirt"] # Added "shirt" as it's common in Fashion-IQ
MODEL_ID = "Salesforce/blip-itm-base-coco"

# --- 2. LOAD SYNTHETIC CAPTIONS (Gemma-3) ---
print(f"📂 Loading synthetic captions from {SYNTHETIC_CAPTIONS_FILE}...")
with open(SYNTHETIC_CAPTIONS_FILE, 'r') as f:
    synth_data = json.load(f)['modified_captions']

# Create a lookup: (image_id, instruction) -> modified_caption
# This allows us to retrieve the exact text for every specific Fashion-IQ query
gemma_lookup = {
    (d['image_id'], d['modification_instruction']): d['modified_caption'] 
    for d in synth_data
}
print(f"Loaded {len(gemma_lookup)} synthetic captions.")

# --- 3. LOAD MODEL ---
print("🚀 Loading BLIP-ITM...")
processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = BlipForImageTextRetrieval.from_pretrained(MODEL_ID, torch_dtype=torch.float16, cache_dir=CACHE_DIR, local_files_only=True).to(DEVICE).eval()

summary_results = {}

# --- 4. CATEGORY LOOP ---
for cat in CATEGORIES:
    print(f"\n--- 🧥 Starting Category: {cat.upper()} ---")
    val_json = os.path.join(CAPTION_DIR, f"cap.{cat}.val.json")
    if not os.path.exists(val_json):
        print(f"Skipping {cat}, file not found.")
        continue
        
    with open(val_json, 'r') as f: data = json.load(f)

    subset = data # You can limit this to [:500] if you want to test quickly
    gallery_ids = list(set([item['target'] for item in subset] + [item['candidate'] for item in subset]))
    valid_gallery = [gid for gid in gallery_ids if os.path.exists(os.path.join(IMAGES_DIR, f"{gid}.jpg"))]
    
    # --- STEP A: PRE-COMPUTE VISION FEATURES ---
    print(f"🖼️  Caching features for {len(valid_gallery)} gallery images...")
    gallery_cache = []
    for g_id in tqdm(valid_gallery):
        img = Image.open(os.path.join(IMAGES_DIR, f"{g_id}.jpg")).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE, dtype=torch.float16)
        with torch.inference_mode():
            img_output = model.vision_model(pixel_values=pixel_values)
            gallery_cache.append(img_output[0])
    
    gallery_features = torch.cat(gallery_cache)

    # --- STEP B: BATCHED RETRIEVAL (USING GEMMA CAPTIONS) ---
    metrics = {"R1": 0, "R5": 0, "R10": 0, "R50": 0}
    processed_count = 0

    print(f"🔍 Ranking {len(subset)} queries...")
    for item in tqdm(subset):
        tgt_id = item['target']
        ref_id = item['candidate']
        instruction = item['captions'][0] # Use the first caption as the key
        
        if not os.path.exists(os.path.join(IMAGES_DIR, f"{tgt_id}.jpg")): continue
        
        # LOOKUP: Get the Gemma-3 generated caption
        text = gemma_lookup.get((ref_id, instruction))
        
        # If we don't have a generated caption for this triplet, skip it
        if not text: continue
        
        text_inputs = processor(text=text, return_tensors="pt").to(DEVICE)
        
        all_scores = []
        for i in range(0, len(gallery_features), BATCH_SIZE):
            batch_features = gallery_features[i : i + BATCH_SIZE]
            curr_bs = batch_features.shape[0]
            
            batch_input_ids = text_inputs.input_ids.repeat(curr_bs, 1)
            batch_attn_mask = text_inputs.attention_mask.repeat(curr_bs, 1)

            with torch.inference_mode():
                outputs = model.text_encoder(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attn_mask,
                    encoder_hidden_states=batch_features,
                    return_dict=True,
                )
                batch_scores = model.itm_head(outputs.last_hidden_state[:, 0, :])
                probs = torch.softmax(batch_scores, dim=-1)[:, 1]
                all_scores.extend(probs.cpu().tolist())

        # Rank and Calculate Recall
        ranked_ids = [valid_gallery[idx] for idx in np.argsort(all_scores)[::-1]]
        if ranked_ids[0] == tgt_id: metrics["R1"] += 1
        if tgt_id in ranked_ids[:5]: metrics["R5"] += 1
        if tgt_id in ranked_ids[:10]: metrics["R10"] += 1
        if tgt_id in ranked_ids[:50]: metrics["R50"] += 1
        processed_count += 1

    summary_results[cat] = {k: (v / processed_count) * 100 for k, v in metrics.items()}
    print(f"✅ {cat.upper()} category completed.")

# --- 5. FINAL SUMMARY TABLE ---
print("\n" + "="*70)
print(f"{'CATEGORY':<12} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'R@50':<8}")
print("-" * 70)
for cat, res in summary_results.items():
    print(f"{cat.upper():<12} | {res['R1']:>6.2f}% | {res['R5']:>6.2f}% | {res['R10']:>6.2f}% | {res['R50']:>6.2f}%")
print("="*70)