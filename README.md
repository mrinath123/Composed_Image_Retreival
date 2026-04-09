# Composed_Image_Retreival

This project moves beyond standard retrieval by using **LVLMs** (Multimodal LLM) to synthesize high-quality, full-sentence descriptions of fashion items. These descriptions are then used to perform zero-shot retrieval using **BLIP-ITM**.

## 🚀 The Pipeline Architecture

Unlike traditional methods that use raw, fragmented human instructions (e.g., "more blue and longer"), our pipeline follows a three-stage transformation process:
1. **Visual Grounding:** Converting raw images into structured base captions.
2. **LLM Transformation:** Blending base captions with human instructions to create a "Target State" description.
3. **Optimized Retrieval:** Matching synthesized text against a pre-computed image gallery.

---

## 📂 File Breakdown & Logic

### 1. `text_fashion.py` (The Dataset Curator)
**What it does:** This script creates a "Closed Set" evaluation sandbox. 
- It scans your local disk to find which images actually exist.
- It filters the Fashion-IQ metadata to ensure that for every query, the **Reference image** and the **Target image** are both physically present.
- It outputs a `closed_set_image_ids.json` (The Gallery) and `closed_set_queries.json` (The Test Set).

### 2. `captn_gen.py` (The Vision Generator)
**What it does:** Uses **Gemma-3**'s multimodal capabilities to "see" the reference image.
- It ignores the human model and focuses strictly on the garment's attributes (color, style, graphics).
- It produces a clean, descriptive "Base Caption" for every image in your closed set.

### 3. `modify_captn.py` (The Relational Editor)
**What it does:** This is the "brain" of the text pipeline. 
- It takes the **Base Caption** and the **Fashion-IQ Instruction** (e.g., "make it red").
- Using Few-Shot prompting, it forces Gemma-3 to perform a logical edit.
- **Output:** A single sentence describing what the *target* image should look like.

### 4. `blip_infer.py` (The Performance Judge)
**What it does:** This is a high-speed retrieval engine using **BLIP-ITM**.
- **Feature Caching:** It encodes all gallery images into vectors first so it doesn't have to re-process them for every query.
- **Batched Scoring:** It compares the synthesized text against the gallery in batches for maximum GPU utilization.
- **Metrics:** It calculates Recall@1, 5, 10, and 50.

---

## 📊 Dataset Setup

1. **Download Fashion-IQ:** Request access at the [Official Repository](https://github.com/XiaoxiaoGuo/fashion-iq).
2. **Structure:**
   ```text
   /fashion-iq/
   └── fashionIQ_dataset/
       ├── images/ (Place all .jpg files here)
       └── captions/ (cap.dress.val.json, cap.shirt.val.json, etc.)
