import os
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re # Import the regular expression library

# --- 1. Configuration ---
print("Starting caption modification script...")

CAPTIONS_FILE = "/BS/DApt/work/fashion-iq/base_captions_v2.json"
FASHION_IQ_DATA_FILE = "/BS/DApt/work/fashion-iq/fashionIQ_dataset/captions/cap.shirt.val.json"
OUTPUT_FILE = "/BS/DApt/work/fashion-iq/modified_captions_short_v2.json"
MODEL_ID = "google/gemma-3-4b-it"

# --- NEW HELPER FUNCTION ---
def clean_base_caption(text):
    """Removes conversational prefixes and newlines to get a clean sentence."""
    # Remove common prefixes
    text = re.sub(r"^(here's a concise description of the t-shirt:|the image shows a|this is a)\s*", "", text, flags=re.IGNORECASE)
    # Replace newlines and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. Load Your Generated Captions ---
print(f"Loading 100 base captions from: {CAPTIONS_FILE}")
try:
    with open(CAPTIONS_FILE, 'r') as f:
        captions_to_modify = json.load(f)
    print(f"Loaded {len(captions_to_modify)} captions.")
except FileNotFoundError:
    print(f"Error: Base captions file not found at {CAPTIONS_FILE}")
    exit()

# --- 3. Load Fashion-IQ Modification Instructions ---
print(f"Loading Fashion-IQ modification texts...")
try:
    with open(FASHION_IQ_DATA_FILE, 'r') as f:
        fashion_iq_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Fashion-IQ data file not found.")
    exit()

# Create a lookup map: {image_id: first_modification_text}
image_to_modification = {}
for triplet in fashion_iq_data:
    ref_id = triplet.get('candidate')
    if ref_id not in image_to_modification and triplet.get('captions'):
        image_to_modification[ref_id] = triplet['captions'][0]
print(f"Created a lookup map with {len(image_to_modification)} modification instructions.")

# --- 4. Load Model ---
print(f"Loading model for text modification: {MODEL_ID}")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 5. Generate 100 New, Modified Captions ---
final_modified_captions = []
for item in tqdm(captions_to_modify, desc="Creating Modified Captions"):
    image_id = item['image_id']
    
    # Clean the original caption before using it
    original_caption_raw = item['caption']
    original_caption = clean_base_caption(original_caption_raw)
    
    modification_text = image_to_modification.get(image_id)
    if not modification_text: continue

    # --- THE NEW, STRONGER PROMPT ---
    # A much stronger prompt that forces the model to edit properly
    prompt = f"""You are a fashion description editor. Your task is to modify a description of a garment based on an instruction. Keep the description as short as possible and avoid describing anthing except the garment. Dont add the originals desciptions colour or graphic in modified. and if the original is shirt, modifed is also shirt just with differnt description.

### EXAMPLE ###
Original Description: "A gray and black striped tank top with Superman graphic."
Instruction: "is red and a t-shirt with 'Kenny Powers' graphic"
Edited Description: "A red t-shirt with a 'Kenny Powers' graphic."

Original Description: "A blue striped t-shirt with Superman graphic."
Instruction: "Has a darker print on front."
Edited Description: "A t-shirt having darker print."

Original Description: "A blue striped tank top."
Instruction: "Has a darker print and plain."
Edited Description: "A tank top having darker print but being plain."

### TASK ###
Original Description: "{original_caption}"
Instruction: "{modification_text}"
Edited Description:
"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        modified_caption = processor.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()

        final_modified_captions.append({
            'image_id': image_id,
            'original_caption': original_caption, # Save the cleaned version for clarity
            'modification_instruction': modification_text,
            'modified_caption': modified_caption
        })

    except Exception as e:
        print(f"\nError processing image {image_id}: {e}")
        continue

# --- 6. Save the Final 100 Modified Captions ---
# ... (This part of the code is perfect and needs no changes)
final_data = {
    "info": {
        "description": "A list of 100 captions, modified based on Fashion-IQ instructions.",
        "total_captions": len(final_modified_captions),
        "model_used": MODEL_ID
    },
    "modified_captions": final_modified_captions
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_data, f, indent=2)

print(f"\n✅ Success! Saved {len(final_modified_captions)} modified captions to {OUTPUT_FILE}")

# Show a sample from the output file to verify
if final_modified_captions:
    print("\n--- Example of a modified caption ---")
    print(json.dumps(final_modified_captions[0], indent=2))