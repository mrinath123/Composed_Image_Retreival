import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re

# --- 1. Configuration ---
print("Setting up similarity scoring and retrieval evaluation...")
DATASET_ROOT = "/BS/DApt/work/fashion-iq/fashionIQ_dataset"
MODIFIED_CAPTIONS_FILE = "/BS/DApt/work/fashion-iq/modified_captions_500_short.json"
VERIFIED_IDS_FILE = "/BS/DApt/work/fashion-iq/closed_set_500_image_ids.json"
FASHION_IQ_DATA_FILE = os.path.join(DATASET_ROOT, "captions", "cap.shirt.val.json")
OUTPUT_FILE = "/BS/DApt/work/fashion-iq/retrieval_results_500_short.json"
VISUALIZATION_FILE = "/BS/DApt/work/fashion-iq/retrieval_visualization_short.jpg"

MODEL_ID = "google/gemma-3-4b-it"

# --- 2. Helper Functions ---
def extract_score(text):
    """Robustly extracts a number from the model's text output."""
    matches = re.findall(r'\d+\.?\d*', text)
    if matches:
        try:
            score = float(matches[0])
            return min(10.0, max(0.0, score))
        except (ValueError, IndexError): return 0.0
    return 0.0

def create_visualization_grid(results, images_dir, output_path, num_samples=5):
    """Creates a JPEG grid summarizing the first few retrieval results."""
    print(f"\nCreating visualization for the first {num_samples} queries...")
    
    THUMB_SIZE, PADDING, BORDER_WIDTH, TEXT_HEIGHT = 150, 10, 4, 80
    IMG_PER_ROW, ROW_HEIGHT = 7, THUMB_SIZE + PADDING + TEXT_HEIGHT
    
    W = (THUMB_SIZE * IMG_PER_ROW) + (PADDING * (IMG_PER_ROW + 1))
    H = ROW_HEIGHT * num_samples
    
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font_header = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        print("Warning: DejaVuSans font not found. Using default font.")
        font_header = ImageFont.load_default()
        font_label = ImageFont.load_default()

    for i, result in enumerate(results[:num_samples]):
        y_offset = i * ROW_HEIGHT
        draw.text((PADDING, y_offset + PADDING), f"Query {i+1}: {result['query_text']}", fill="black", font=font_header, width=W-PADDING)
        image_map = {
            'ref': Image.open(os.path.join(images_dir, f"{result['reference_id']}.jpg")).resize((THUMB_SIZE, THUMB_SIZE)),
            'gt': Image.open(os.path.join(images_dir, f"{result['ground_truth_target_id']}.jpg")).resize((THUMB_SIZE, THUMB_SIZE))
        }
        for j, item in enumerate(result['top_5_retrieved']):
            image_map[f'ret_{j}'] = Image.open(os.path.join(images_dir, f"{item['id']}.jpg")).resize((THUMB_SIZE, THUMB_SIZE))
        
        images_to_draw = [('Reference', image_map['ref'], None), ('Ground Truth', image_map['gt'], None)] + \
                         [(f"Rank {j+1} (Score: {item['score']:.1f})", image_map[f'ret_{j}'], item['id'] == result['ground_truth_target_id']) 
                          for j, item in enumerate(result['top_5_retrieved'])]

        for j, (label, img, is_correct) in enumerate(images_to_draw):
            x_offset = PADDING + j * (THUMB_SIZE + PADDING)
            border_color = 'lightgray'
            if is_correct is True: border_color = 'green'
            if is_correct is False: border_color = 'red'
            border = Image.new('RGB', (THUMB_SIZE + 2 * BORDER_WIDTH, THUMB_SIZE + 2 * BORDER_WIDTH), border_color)
            border.paste(img, (BORDER_WIDTH, BORDER_WIDTH))
            canvas.paste(border, (x_offset, y_offset + TEXT_HEIGHT // 2))
            draw.text((x_offset, y_offset + TEXT_HEIGHT // 2 + THUMB_SIZE + PADDING), label, fill="black", font=font_label)
    
    canvas.save(output_path)
    print(f"✅ Early visualization saved to {output_path}")


# --- 3. Load Data ---
print("Loading necessary data files...")
with open(MODIFIED_CAPTIONS_FILE, 'r') as f: queries = json.load(f)['modified_captions']
with open(VERIFIED_IDS_FILE, 'r') as f: candidate_image_ids = json.load(f)
with open(FASHION_IQ_DATA_FILE, 'r') as f: fashion_iq_triplets = json.load(f)
target_lookup = {(triplet['candidate'], mod_text): triplet['target'] for triplet in fashion_iq_triplets for mod_text in triplet['captions']}
print(f"Loaded {len(queries)} total queries against a gallery of {len(candidate_image_ids)} images.")

# --- 3.5. FILTER TO ONLY VALID QUERIES ---
print("\n🔍 FILTERING TO ONLY VALID QUERIES (GT in gallery)...")
candidate_set = set(candidate_image_ids)
valid_queries = []
skipped_queries = 0

for query in queries:
    reference_id = query['image_id']
    mod_instruction = query['modification_instruction']
    ground_truth_target_id = target_lookup.get((reference_id, mod_instruction))
    
    if ground_truth_target_id and ground_truth_target_id in candidate_set:
        valid_queries.append(query)
    else:
        skipped_queries += 1

print(f"📊 FILTERING RESULTS:")
print(f"   Total queries: {len(queries)}")
print(f"   Valid queries (GT in gallery): {len(valid_queries)}")
print(f"   Skipped queries (GT missing): {skipped_queries}")
print(f"✅ Proceeding with {len(valid_queries)} valid queries only!")

# --- 4. Load Model ---
print(f"\nLoading model: {MODEL_ID}")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 5. Perform N x N Similarity Scoring with Incremental Updates ---
images_dir = os.path.join(DATASET_ROOT, "images")
all_query_results = []
visualization_generated = False # Flag to ensure we only generate the image once

# Outer loop: Iterate through each of our VALID queries only
for i, query in enumerate(tqdm(valid_queries, desc="Processing Valid Queries")):
    query_text = query['modified_caption']
    reference_id = query['image_id']
    mod_instruction = query['modification_instruction']
    ground_truth_target_id = target_lookup.get((reference_id, mod_instruction))
    
    # We already verified this is valid, but double-check
    if not ground_truth_target_id or ground_truth_target_id not in candidate_set:
        print(f"WARNING: Query {i} unexpectedly invalid, skipping...")
        continue

    scores = []
    # Inner loop: Score against all candidates
    for candidate_id in tqdm(candidate_image_ids, desc=f"Valid Query {i+1}/{len(valid_queries)}", leave=False):
        image_path = os.path.join(images_dir, f"{candidate_id}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f'On a scale of 0 to 10, how well does the image match the following description?\nDescription: "{query_text}"\nProvide only a single numerical score.\nScore:'
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            response_text = processor.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
            score = extract_score(response_text)
            scores.append(score)
        except Exception as e:
            print(f"Error scoring candidate {candidate_id}: {e}")
            scores.append(0.0)
    
    # Get top 5 retrieved for detailed analysis
    sorted_indices_top5 = np.argsort(scores)[::-1][:5]
    top_5_retrieved = [{"rank": r + 1, "id": candidate_image_ids[idx], "score": scores[idx]} for r, idx in enumerate(sorted_indices_top5)]
    
    # Append the result for this query to our list in memory
    all_query_results.append({
        "query_text": query_text,
        "reference_id": reference_id,
        "ground_truth_target_id": ground_truth_target_id,
        "scores": scores,
        "top_5_retrieved": top_5_retrieved
    })

    # === INCREMENTAL SAVING AND EARLY VISUALIZATION (same as before) ===
    # After each query, overwrite the JSON file with the latest results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"details": all_query_results}, f, indent=2)

    # After the 5th valid query is done, generate the visualization
    if (i + 1) == 5 and not visualization_generated:
        create_visualization_grid(all_query_results, images_dir, VISUALIZATION_FILE)
        visualization_generated = True # Set flag to true

# --- 6. Calculate Final Metrics ---
if not all_query_results:
    print("No valid queries were processed. Exiting.")
    exit()

r_at_1, r_at_10, r_at_50 = 0, 0, 0
num_queries = len(all_query_results)

for result in all_query_results:
    scores = np.array(result['scores'])
    candidates = np.array(candidate_image_ids)
    ground_truth_id = result['ground_truth_target_id']
    sorted_indices = np.argsort(scores)[::-1]
    ranked_candidates = candidates[sorted_indices]
    
    if ranked_candidates[0] == ground_truth_id: r_at_1 += 1
    if ground_truth_id in ranked_candidates[:10]: r_at_10 += 1
    if ground_truth_id in ranked_candidates[:50]: r_at_50 += 1

print("\n--- Retrieval Results ---")
print(f"Total Valid Queries Processed: {num_queries}")
print(f"Recall@1:  {r_at_1 / num_queries:.2%} ({r_at_1}/{num_queries})")
print(f"Recall@10: {r_at_10 / num_queries:.2%} ({r_at_10}/{num_queries})")
print(f"Recall@50: {r_at_50 / num_queries:.2%} ({r_at_50}/{num_queries})")

# --- 7. Save Final Detailed Results (with metrics) ---
with open(OUTPUT_FILE, 'w') as f:
    json.dump({
        "metrics": {
            "total_valid_queries": num_queries,
            "total_skipped_queries": skipped_queries,
            "R@1": r_at_1 / num_queries,
            "R@10": r_at_10 / num_queries,
            "R@50": r_at_50 / num_queries,
        },
        "details": all_query_results
    }, f, indent=2)
print(f"\n✅ Final results (with metrics) saved to {OUTPUT_FILE}")
print(f"🎯 Summary: Processed {num_queries} valid queries, skipped {skipped_queries} invalid ones")