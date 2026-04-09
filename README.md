# 👕 Composed_Image_Retreival

This repository contains a pipeline for composed image retrieval task. It utilizes LVLMs like **Gemma-3** to synthesize high-quality training data and **BLIP** (Vision-Language Model) to perform zero-shot and fine-tuned retrieval.

## 🚀 Overview
Traditional fashion retrieval uses fragmented instructions (e.g., "more blue"). This pipeline transforms those fragments into full, descriptive target sentences using Gemma-3, significantly improving the alignment between text queries and target images.

---

## 📂 Project Structure & Script Guide

### 🧱 Phase 1: Data Curation
* **`check_ref.py`**: **The Entry Point.** Scans your local disk for images and filters the Fashion-IQ metadata. It creates a "Closed Set" gallery and valid query list to ensure 100% path integrity during evaluation.

### ✍️ Phase 2: Synthetic Captioning (Gemma-3)
* **`captn_gen.py`**: Vision-to-Text. Uses Gemma-3 to describe reference images, focusing purely on garment attributes (color, style, fabric) while ignoring the person.
* **`modify_captn.py`**: Relational Editing. Takes the base caption and applies the human instruction (e.g., "add stripes") to generate a final sentence describing the *target* image.

### 🏋️ Phase 3: Model Training
* **`blip_train.py`**: Training an **Adapter Head**. Keeps the BLIP backbone frozen and trains a linear layer to specialize the model on fashion-specific vocabulary and features.

### 🔍 Phase 4: Retrieval & Inference
* **`run_ret_base.py`**: A baseline script for evaluating standard BLIP models without synthetic captions or training.
* **`blip_infer.py`**: Evaluation script for the trained Adapter model. Calculates Recall@K (1, 5, 10).
* **`blip_fast_retrieval.py`**: **Optimized Search.** Uses Feature Caching and Batching to rank thousands of images in seconds rather than hours.

### 🧪 Phase 5: Advanced Diffusion Experiments
* **`diffusion_img.py`**: Uses **InstructPix2Pix** to "dream" the target image based on the reference and instruction, then performs an image-to-image search.
* **`diff_text.py`**: Fusion Retrieval. Combines the vectors of the "Dreamed Image" and the "Modified Text" for the most accurate retrieval results.

---

## 🛠 Setup & Installation

1. Clone the repository:
    git clone <your-repo-link>
    cd fashion-iq-pipeline

2. Install dependencies:
    pip install -r requirements.txt

3. Dataset Preparation:

Place the Fashion-IQ dataset in the following structure:

fashionIQ_dataset/
├── images/   # All .jpg files
└── captions/ # cap.dress.val.json, cap.shirt.val.json, etc.

---

## 🏃 Execution Workflow

Run the pipeline in this exact order:

1. Initialize the Gallery
    # Set TARGET_SET_SIZE = None in the script for full dataset evaluation
    python check_ref.py

2. Synthesize Training Data
    python captn_gen.py
    python modify_captn.py

3. (Optional) Train the Adapter
    python blip_train.py

4. Perform Retrieval Evaluation
    python blip_fast_retrieval.py

---

## 📊 Evaluation Metrics

- Recall@1: Is the correct item the #1 result?
- Recall@10: Is the correct item in the top 10?
- Recall@50: Is the correct item in the top 50?

---

## 💡 Important Configuration

- Cache: Default path is /BS/DApt/work/huggingface_cache. Update CACHE_DIR if needed.
- GPU: Gemma-3 requires high VRAM. If you hit OOM errors:
    - Set torch_dtype=torch.bfloat16
    - Reduce BATCH_SIZE

---

## 📌 Notes

- requirements.txt: Place in project root and run:
    pip install -r requirements.txt

- README.md: This file explains the pipeline and execution order.
