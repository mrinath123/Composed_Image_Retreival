import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval

# --- 1. CONFIGURATION ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR

DATASET_ROOT = "/BS/DApt/work/fashion-iq/fashionIQ_dataset"
MODIFIED_CAPTIONS_FILE = "/BS/DApt/work/fashion-iq/modified_captions_short.json"
VERIFIED_IDS_FILE = "/BS/DApt/work/fashion-iq/closed_set_image_ids.json"
FASHION_IQ_VAL_FILE = os.path.join(DATASET_ROOT, "captions", "cap.shirt.val.json")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
ADAPTER_WEIGHTS = "/BS/DApt/work/fashion-iq/fashioniq_adapter_head.pth"

MODEL_ID = "Salesforce/blip-itm-base-coco"
DEVICE = "cuda"
OUTPUT_FILE = "/BS/DApt/work/fashion-iq/retrieval_results_adapter_final.json"

# --- 2. ARCHITECTURE ---
class BlipLinearAdapter(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.blip_itm = BlipForImageTextRetrieval.from_pretrained(model_id, cache_dir=CACHE_DIR)
        for param in self.blip_itm.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.blip_itm.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        multi_outputs = self.blip_itm.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, return_dict=True
        )
        last_hidden_state = multi_outputs.last_hidden_state 
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        pooled = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.classifier(pooled)

# --- 3. LOAD DATA & MODEL ---
print(f"Loading files from {DATASET_ROOT}...")
with open(MODIFIED_CAPTIONS_FILE, 'r') as f: queries = json.load(f)['modified_captions']
with open(VERIFIED_IDS_FILE, 'r') as f: candidate_image_ids = json.load(f)
with open(FASHION_IQ_VAL_FILE, 'r') as f: fashion_iq_triplets = json.load(f)

target_lookup = {(t['candidate'], c): t['target'] for t in fashion_iq_triplets for c in t['captions']}
candidate_set = set(candidate_image_ids)

model = BlipLinearAdapter(MODEL_ID).to(DEVICE)
model.classifier.load_state_dict(torch.load(ADAPTER_WEIGHTS))
model.eval()
processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

# --- 4. RETRIEVAL LOOP ---
all_query_results = []
valid_queries = [q for q in queries if target_lookup.get((q['image_id'], q['modification_instruction'])) in candidate_set]

print(f"Starting retrieval for {len(valid_queries)} valid queries...")

for query in tqdm(valid_queries, desc="Total Progress"):
    query_text = query['modified_caption']
    gt_id = target_lookup.get((query['image_id'], query['modification_instruction']))

    scores = []
    # Loop over all candidates
    for cand_id in candidate_image_ids:
        try:
            img_path = os.path.join(IMAGES_DIR, f"{cand_id}.jpg")
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=query_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                logits = model(**inputs)
                scores.append(torch.sigmoid(logits).item())
        except:
            scores.append(-1.0)

    # Sort and take top 10
    sorted_idx = np.argsort(scores)[::-1]
    top_10 = [{"id": candidate_image_ids[idx], "score": scores[idx]} for idx in sorted_idx[:10]]
    
    all_query_results.append({"query": query_text, "gt": gt_id, "top_10": top_10})

# --- 5. FINAL METRICS & SAVE ---
r1, r5, r10 = 0, 0, 0
for res in all_query_results:
    top_ids = [item['id'] for item in res['top_10']]
    if top_ids[0] == res['gt']: r1 += 1
    if res['gt'] in top_ids[:5]: r5 += 1
    if res['gt'] in top_ids[:10]: r10 += 1

num = len(all_query_results)
metrics = {"R@1": r1/num, "R@5": r5/num, "R@10": r10/num}

print(f"\n--- Final Metrics ---")
print(f"R@1: {metrics['R@1']:.2%}")
print(f"R@5: {metrics['R@5']:.2%}")
print(f"R@10: {metrics['R@10']:.2%}")

with open(OUTPUT_FILE, 'w') as f:
    json.dump({"metrics": metrics, "results": all_query_results}, f, indent=2)

print(f"✅ Results saved to {OUTPUT_FILE}")