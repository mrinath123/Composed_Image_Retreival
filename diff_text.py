import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForImageTextRetrieval
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# --- 1. ENVIRONMENT ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1" 
DEVICE = "cuda"

BLIP_MODEL = "Salesforce/blip-itm-base-coco"
DIFF_MODEL = "timbrooks/instruct-pix2pix"

# Paths
IMAGES_DIR = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/images"
GALLERY_JSON = "/BS/DApt/work/fashion-iq/closed_set_500_image_ids.json"
QUERIES_JSON = "/BS/DApt/work/fashion-iq/modified_captions_500_short.json"
VAL_JSON = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/captions/cap.shirt.val.json"

# --- 2. LOAD MODELS ---
print("🚀 Loading Diffusion Model (InstructPix2Pix)...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    DIFF_MODEL, torch_dtype=torch.float16, safety_checker=None, cache_dir=CACHE_DIR, local_files_only=True
).to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("🚀 Loading Vision-Language Model (BLIP)...")
# We load the full retrieval model to get the joint text/image projection spaces
blip_proc = AutoProcessor.from_pretrained(BLIP_MODEL, cache_dir=CACHE_DIR, local_files_only=True)
blip_model = BlipForImageTextRetrieval.from_pretrained(BLIP_MODEL, cache_dir=CACHE_DIR, local_files_only=True).to(DEVICE).eval()

# --- 3. HELPER: JOINT EMBEDDING EXTRACTOR ---
def get_blip_emb(img=None, text=None):
    """
    Extracts embeddings in the JOINT multimodal space.
    This allows us to mathematically add Image and Text vectors together.
    """
    with torch.inference_mode():
        if img is not None:
            inputs = blip_proc(images=img, return_tensors="pt").to(DEVICE)
            # Pass through vision model AND vision projection layer
            img_embeds = blip_model.vision_model(**inputs).last_hidden_state[:, 0, :]
            feat = blip_model.vision_proj(img_embeds)
            
        elif text is not None:
            inputs = blip_proc(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            # Pass through text model AND text projection layer
            text_embeds = blip_model.text_encoder(**inputs).last_hidden_state[:, 0, :]
            feat = blip_model.text_proj(text_embeds)
            
        return F.normalize(feat, p=2, dim=1)

# --- 4. PREPARE DATA ---
with open(GALLERY_JSON, 'r') as f: gallery_ids = json.load(f)
with open(QUERIES_JSON, 'r') as f: queries = json.load(f)['modified_captions']
with open(VAL_JSON, 'r') as f: val_data = json.load(f)
target_map = {(t['candidate'], t['captions'][0]): t['target'] for t in val_data}

# --- 5. EMBED GALLERY ---
print("📊 Embedding Gallery Images...")
g_embs_blip = []
for g_id in tqdm(gallery_ids):
    img = Image.open(os.path.join(IMAGES_DIR, f"{g_id}.jpg")).convert("RGB")
    g_embs_blip.append(get_blip_emb(img=img))
g_embs_blip = torch.cat(g_embs_blip)

# --- 6. RETRIEVAL: FUSING DIFFUSION + TEXT ---
print("🎨 Running Experiment: Dreamed Image + Text Fusion...")
q_embs_fused = []
gt_list = []

# Hyperparameter from WeiMoCIR: How much weight to give the text vs the image.
# Since the dreamed image is already heavily modified, we use a smaller alpha (e.g., 0.2 to 0.4)
# just to correct any missing semantic details.
ALPHA = 0.3 

for q in tqdm(queries):
    gt = target_map.get((q['image_id'], q['modification_instruction']))
    if gt and gt in gallery_ids:
        orig_img = Image.open(os.path.join(IMAGES_DIR, f"{q['image_id']}.jpg")).convert("RGB")
        mod_text = q['modification_instruction']
        
        # 1. GENERATE THE "DREAM" IMAGE
        with torch.inference_mode():
            gen_img = pipe(
                mod_text, 
                image=orig_img, 
                num_inference_steps=20, 
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
        
        # 2. EXTRACT SEPARATE EMBEDDINGS
        v_dream = get_blip_emb(img=gen_img)
        t_mod = get_blip_emb(text=mod_text)
        
        # 3. THE MAGIC: COMBINE THEM
        # Equation: q = (1 - alpha) * Dreamed_Image + (alpha) * Modified_Text
        q_fused = ((1 - ALPHA) * v_dream) + (ALPHA * t_mod)
        
        # Must re-normalize after addition
        q_fused = F.normalize(q_fused, p=2, dim=1) 
        
        q_embs_fused.append(q_fused)
        gt_list.append(gt)

q_embs_fused = torch.cat(q_embs_fused)

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

fused_res = get_metrics(q_embs_fused, g_embs_blip, gt_list, gallery_ids)

print("\n" + "="*50)
print(f"{'METRIC':<10} | {'BLIP (Dream + Text Fusion)'}")
print("-" * 50)
for k in [1, 5, 10, 50]:
    print(f"Recall@{k:<2} | {fused_res[k]:.2%}")