import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, BlipProcessor, BlipForImageTextRetrieval
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# --- 1. ENVIRONMENT ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1" 
DEVICE = "cuda"

DINO_MODEL = "facebook/dinov2-base"
BLIP_MODEL = "Salesforce/blip-itm-base-coco"
DIFF_MODEL = "timbrooks/instruct-pix2pix"

# Paths
IMAGES_DIR = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/images"
GALLERY_JSON = "/BS/DApt/work/fashion-iq/closed_set_image_ids.json"
QUERIES_JSON = "/BS/DApt/work/fashion-iq/modified_captions_short.json"
VAL_JSON = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/captions/cap.shirt.val.json" #dress/tee

# --- 2. LOAD MODELS ---
print("🚀 Loading Diffusion Model (InstructPix2Pix)...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    DIFF_MODEL, torch_dtype=torch.float16, safety_checker=None, cache_dir=CACHE_DIR, local_files_only=True
).to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("🚀 Loading Vision Encoders (Corrected Weights)...")
dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL, cache_dir=CACHE_DIR, local_files_only=True)
dino_model = AutoModel.from_pretrained(DINO_MODEL, cache_dir=CACHE_DIR, local_files_only=True).to(DEVICE).eval()

blip_proc = BlipProcessor.from_pretrained(BLIP_MODEL, cache_dir=CACHE_DIR, local_files_only=True)
# FIX: Load the full retrieval model then extract vision_model to avoid "MISSING" keys
full_blip = BlipForImageTextRetrieval.from_pretrained(BLIP_MODEL, cache_dir=CACHE_DIR, local_files_only=True)
blip_model = full_blip.vision_model.to(DEVICE).eval()

# --- 3. HELPER ---
def get_emb(img, mode):
    with torch.inference_mode():
        if mode == "dino":
            inputs = dino_proc(images=img, return_tensors="pt").to(DEVICE)
            out = dino_model(**inputs).last_hidden_state[:, 0, :]
        else:
            inputs = blip_proc(images=img, return_tensors="pt").to(DEVICE)
            out = blip_model(**inputs).last_hidden_state[:, 0, :]
        return F.normalize(out, p=2, dim=1)

# --- 4. PREPARE DATA ---
with open(GALLERY_JSON, 'r') as f: gallery_ids = json.load(f)
with open(QUERIES_JSON, 'r') as f: queries = json.load(f)['modified_captions']
with open(VAL_JSON, 'r') as f: val_data = json.load(f)
target_map = {(t['candidate'], t['captions'][0]): t['target'] for t in val_data}

# --- 5. EMBED GALLERY ---
print("📊 Embedding Gallery Images (Map)...")
g_embs_dino, g_embs_blip = [], []
for g_id in tqdm(gallery_ids):
    img = Image.open(os.path.join(IMAGES_DIR, f"{g_id}.jpg")).convert("RGB")
    g_embs_dino.append(get_emb(img, "dino"))
    g_embs_blip.append(get_emb(img, "blip"))
g_embs_dino = torch.cat(g_embs_dino)
g_embs_blip = torch.cat(g_embs_blip)

# --- 6. RETRIEVAL WITH DIFFUSION ---
print("🎨 Running Proposed Experiment (Diffusion -> Search)...")
q_embs_dino, q_embs_blip, gt_list = [], [], []

for q in tqdm(queries):
    gt = target_map.get((q['image_id'], q['modification_instruction']))
    # Only process if ground truth is in our set
    if gt and gt in gallery_ids:
        orig_img = Image.open(os.path.join(IMAGES_DIR, f"{q['image_id']}.jpg")).convert("RGB")
        
        # GENERATE THE "DREAM" IMAGE
        with torch.inference_mode():
            gen_img = pipe(
                q['modification_instruction'], 
                image=orig_img, 
                num_inference_steps=20, 
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
        
        # EMBED THE GENERATED IMAGE
        q_embs_dino.append(get_emb(gen_img, "dino"))
        q_embs_blip.append(get_emb(gen_img, "blip"))
        gt_list.append(gt)

q_embs_dino = torch.cat(q_embs_dino)
q_embs_blip = torch.cat(q_embs_blip)

# --- 7. METRICS ---
def get_metrics(q_e, g_e, gts, g_list):
    sims = torch.matmul(q_e, g_e.T)
    ranks = torch.argsort(sims, descending=True, dim=1).cpu().numpy()
    results = {1: 0, 5: 0, 10: 0, 50: 0}
    for i, gt in enumerate(gts):
        top_k = [g_list[idx] for idx in ranks[i]]
        for k in results.keys():
            if gt in top_k[:k]: results[k] += 1
    return {k: v / len(gts) for k, v in results.items()}

dino_res = get_metrics(q_embs_dino, g_embs_dino, gt_list, gallery_ids)
blip_res = get_metrics(q_embs_blip, g_embs_blip, gt_list, gallery_ids)

print("\n" + "="*50)
print(f"{'METRIC':<10} | {'DINOv2 (Proposed)':<18} | {'BLIP (Proposed)'}")
print("-" * 50)
for k in [1, 5, 10, 50]:
    print(f"Recall@{k:<2} | {dino_res[k]:.2%} | {blip_res[k]:.2%}")