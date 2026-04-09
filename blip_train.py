import os
import sys
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt

# --- 0. ENVIRONMENT & PATH SETUP ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# Define the output directory clearly
OUTPUT_DIR = "/BS/DApt/work/fashion-iq"
DATASET_ROOT = "/BS/DApt/work/fashion-iq/fashionIQ_dataset"
TRAIN_JSON = os.path.join(DATASET_ROOT, "captions", "cap.shirt.train.json")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
VERIFIED_IDS_FILE = "/BS/DApt/work/fashion-iq/closed_set_image_ids.json"

MODEL_ID = "Salesforce/blip-itm-base-coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
TOTAL_ITERS = 100  # Exact number of training steps
LEARNING_RATE = 1e-4

print(f"--- Environment Setup ---")
print(f"✅ Cache directory: {CACHE_DIR}")
print(f"✅ Output directory: {OUTPUT_DIR}")
print(f"✅ Device: {DEVICE}")

# --- 1. DATASET CLASS ---
class FashionIQTrainDataset(Dataset):
    def __init__(self, json_file, exclude_ids_file, processor):
        self.processor = processor
        with open(exclude_ids_file, 'r') as f:
            exclude_ids = set(json.load(f))
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        self.data = [item for item in raw_data if item['target'] not in exclude_ids]
        self.all_train_ids = [item['target'] for item in self.data]
        print(f"Final training pool: {len(self.data)} items (Filtered).")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        combined_caption = f"{item['captions'][0]} and {item['captions'][1]}"
        pos_id, neg_id = item['target'], random.choice(self.all_train_ids)
        while neg_id == pos_id: neg_id = random.choice(self.all_train_ids)
        try:
            pos_img = Image.open(os.path.join(IMAGES_DIR, f"{pos_id}.jpg")).convert("RGB")
            neg_img = Image.open(os.path.join(IMAGES_DIR, f"{neg_id}.jpg")).convert("RGB")
            pos_in = self.processor(images=pos_img, text=combined_caption, return_tensors="pt", padding="max_length", truncation=True)
            neg_in = self.processor(images=neg_img, text=combined_caption, return_tensors="pt", padding="max_length", truncation=True)
            return {"pos": {k: v.squeeze(0) for k, v in pos_in.items()}, "neg": {k: v.squeeze(0) for k, v in neg_in.items()}}
        except: return self.__getitem__(random.randint(0, len(self.data)-1))

# --- 2. THE ADAPTER ARCHITECTURE ---
class BlipLinearAdapter(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.blip_itm = BlipForImageTextRetrieval.from_pretrained(model_id, cache_dir=CACHE_DIR)
        for param in self.blip_itm.parameters(): param.requires_grad = False
        self.classifier = nn.Linear(768, 1)

    def forward(self, pixel_values, input_ids, attention_mask, verbose=False):
        vision_outputs = self.blip_itm.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        multi_outputs = self.blip_itm.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, return_dict=True)
        last_hidden_state = multi_outputs.last_hidden_state 
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        pooled_features = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.classifier(pooled_features)

# --- 3. TRAINING SETUP ---
processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
dataset = FashionIQTrainDataset(TRAIN_JSON, VERIFIED_IDS_FILE, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BlipLinearAdapter(MODEL_ID).to(DEVICE)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# --- 4. ITERATION-WISE TRAINING LOOP ---
print(f"\n--- Starting Training for {TOTAL_ITERS} Iterations ---")
model.train()
iteration_losses = []
data_iter = iter(dataloader)

# Setup TQDM for progress tracking
pbar = tqdm(range(1, TOTAL_ITERS + 1), desc="Training Progress", unit="it")

for i in pbar:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    optimizer.zero_grad()
    
    # Positive and Negative passes
    pos_data = {k: v.to(DEVICE) for k, v in batch['pos'].items()}
    pos_logits = model(**pos_data).squeeze(-1)
    neg_data = {k: v.to(DEVICE) for k, v in batch['neg'].items()}
    neg_logits = model(**neg_data).squeeze(-1)

    loss = criterion(pos_logits, torch.ones_like(pos_logits)) + criterion(neg_logits, torch.zeros_like(neg_logits))
    loss.backward()
    optimizer.step()

    iter_loss = loss.item()
    iteration_losses.append(iter_loss)
    
    # Update progress bar with current loss
    pbar.set_postfix({"loss": f"{iter_loss:.4f}"})

# --- 5. PLOTTING, LOGGING & SAVING ---
# 1. Save the Loss Figure
plot_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
plt.figure(figsize=(10, 5))
plt.plot(range(1, TOTAL_ITERS + 1), iteration_losses, color='blue', label='BCE Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Training Loss over {TOTAL_ITERS} Iterations')
plt.grid(True)
plt.legend()
plt.savefig(plot_path)
print(f"\n✅ Loss plot saved to: {plot_path}")

# 2. Save the Weights
weights_path = os.path.join(OUTPUT_DIR, "fashioniq_adapter_head.pth")
torch.save(model.classifier.state_dict(), weights_path)
print(f"✅ Adapter weights saved to: {weights_path}")

print("\n--- Training Summary ---")
print(f"Initial Loss: {iteration_losses[0]:.4f}")
print(f"Final Loss: {iteration_losses[-1]:.4f}")