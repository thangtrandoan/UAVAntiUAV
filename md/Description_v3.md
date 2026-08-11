# COMPLETE PROBLEM STATEMENT & CODE IMPLEMENTATION SPECIFICATION

## 1. Project Context & Objectives
- **Project Name:** Air-to-Air UAV Re-Identification under Out-of-View & Re-appearance Scenarios.
- **Core Task:** In an aerial pursuit scenario (Air-to-Air UAV Tracking), a pursuer UAV tracks target UAVs. When target UAVs disappear from the Field-of-View (Out-of-View / Full Occlusion) and reappear under severe viewpoint/scale changes and motion blur, the system must accurately **Re-Identify (ReID)** the original target IDs from a pool of candidates and distractors.
- **Key Architectural Focus:** Real-time Video-based ReID using a Pre-trained Visual Backbone + Vision Mamba Temporal Memory Engine + Multi-Task Joint Loss.

---

## 2. Model Architecture Specifications

### A. Pre-trained Visual Backbone
- **Weights Source:** Pre-trained checkpoint on the VRU (Vehicle Re-identification from UAV) dataset.
- **Architecture:** Hierarchical Vision Transformer (HiViT-Base or Swin-Transformer) matching the VRU checkpoint.
- **Role:** Extracts fine-grained multi-scale spatial features (frame structure, rotor layout, color/surface patterns) from cropped UAV patches.

### B. Mamba Temporal Memory Engine
- **Architecture:** Selective State Space Model (Vision Mamba / Vim block with 1D Unidirectional Scanning).
- **Role:** Processes a sequence of strided frame feature vectors before target disappearance ($T_{before}$), compressing historical appearance and motion cues into a compact **Temporal Token / Memory Vector** with $\mathcal{O}(N)$ linear complexity.

---

## 3. Data Strategy & Smart Mining (`data_miner.py`)

### A. Strided Sampling Protocol
- **Sequence Length ($N$):** $N = 16$ frames per sequence.
- **Sampling Stride ($Stride = 3$):** Select 16 frames with a stride of 3 frames (covering $\sim 1.6s$ of historical context at 30 FPS) to eliminate frame-to-frame redundancy.
- **Anchor-First Boundary Alignment:**
  - $T_{before}$ Sequence: Fix the boundary frame $T_{disappear}$ (last frame before `OV`/`FO`) as the anchor, and sample 15 preceding frames with $Stride = 3$.
  - $T_{after}$ Sequence: Fix the boundary frame $T_{reappear}$ (first frame post-appearance) as the anchor, and sample 15 succeeding frames with $Stride = 3$.

### B. Smart Mining & Synthetic Augmentation
- Parse `UAV-Anti-UAV` dataset annotations for attribute tags: `OV` (Out-of-View), `FO` (Full Occlusion), `SD` (Similar Distractors), and `SO` (Small Object).
- Extract hard negative distractor crops from sequences labeled with `SD` as well as external datasets (`Anti-UAV318` / `Anti-UAV600`).
- Apply **Context-Enriched Cropping** ($15\%$ margin/padding added around target Bounding Boxes).
- Apply **Synthetic Disappearance & Masking** (Random Erasing, Motion Blur, Scale Jitter) to augment $T_{before} \leftrightarrow T_{after}$ pairs.

---

## 4. Loss Functions & Fine-Tuning Setup

### A. Multi-Task Joint Loss
$$\mathcal{L}_{total} = \mathcal{L}_{ID} + \lambda_1 \mathcal{L}_{Hard-Triplet} + \lambda_2 \mathcal{L}_{Temporal\_Consistency} + \lambda_3 \mathcal{L}_{Center}$$

1. **$\mathcal{L}_{ID}$ (Cross-Entropy Loss with Label Smoothing = 0.1):** Global class discriminability.
2. **$\mathcal{L}_{Hard-Triplet}$ (Hard-Margin Triplet Loss):** Evaluated on $L_2$-normalized feature vectors between $T_{before}$ (Anchor), $T_{after}$ (Positive), and Distractors (Negative).
3. **$\mathcal{L}_{Center}$ (Center Loss):** Pulls all variations toward the centroid of the target ID. Applied on $L_2$-normalized feature vectors ($\lambda_3 = 0.05$).
4. **$\mathcal{L}_{Temporal\_Consistency}$ ($\lambda_2 = 0.1 - 0.2$):** Constrains Mamba latent memory state transitions across strided frames.

### B. Fine-Tuning Optimization Techniques
- **Discriminative Learning Rates:** Backbone $LR = 1 \times 10^{-5}$, Mamba Engine & ReID Heads $LR = 1 \times 10^{-4}$.
- **Backbone Freezing:** Freeze Visual Backbone for the first 5 Warm-up Epochs; unfreeze for end-to-end training afterward.
- **Identity-Aware Sampler ($P \times K$ Sampler):** $P = 8$ IDs per batch, $K = 4$ strided sequences per ID (Batch Size = 32).
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
- **Mixed Precision:** Use `torch.cuda.amp.autocast()` for VRAM efficiency.

---

## 5. Multi-Object Evaluation Protocol (`evaluate.py`)

Evaluate on 3 scenarios: (1) Short Disappearance ($<5s$), (2) Long Disappearance ($>15s$), and (3) Re-appearance with Distractors.

### A. Multi-Object Query & Global Gallery Structure
- **Global Gallery ($G$ items):** For a video containing $M$ target UAVs, extract strided 16-frame sequences ($Stride = 3$) for ALL targets at $T_{before}$ + $K$ Distractors.
  `Gallery = [Seq_Tbefore_ID1, Seq_Tbefore_ID2, ..., Seq_Tbefore_IDM] + Distractor_Pool`
- **Individual Queries ($Q$ items):** Each target reappearing at $T_{after}$ forms an independent Query sequence ($Stride = 3$).
  `Queries = [Seq_Tafter_ID1, Seq_Tafter_ID2, ..., Seq_Tafter_IDM]`

### B. Matrix Matching & Offline Metrics
- Compute Gallery Feature Matrix: $F_{Gallery} \in \mathbb{R}^{G \times D}$
- Compute Query Feature Matrix: $F_{Query} \in \mathbb{R}^{Q \times D}$
- Distance Matrix: $S = F_{Query} \times F_{Gallery}^T \quad \in \mathbb{R}^{Q \times G}$ (Cosine Similarity)
- Metrics: **Rank-1 Accuracy (%)**, **mAP (%)**, **mINP (%)**.

### C. Online Real-Time Simulation
- Simulate video stream post-reappearance ($Stride = 3$ ingestion into Mamba Engine).
- Perform candidate detection $\rightarrow$ Batch GPU Feature Extraction $\rightarrow$ Matrix Match.
- Metrics: **Re-acquisition Latency (ms/frames to issue confirmed match)** and **False Alarm Rate (%)**.

---

## 6. Real-Time Inference Logic

1. **State 1: Tracking Mode**
   - Light Tracker runs at 30+ FPS.
   - Every 3 frames ($Stride = 3$), feed the target crop into the Mamba Memory Engine to update the `Temporal Token`.
2. **State 2: Lost Mode ($Confidence < Threshold_{track}$)**
   - Target disappears (`OV` or `FO`).
   - Freeze/Lock the Mamba `Temporal Token` state.
3. **State 3: ReID Verification Mode (Candidate Detection)**
   - Detector finds $M$ candidates in frame.
   - Stack candidate crops into a single Tensor Batch `[M, 3, H, W]` for parallel GPU extraction.
   - Compute Cosine Distance Matrix against the locked `Temporal Token`.
   - If $\max(\text{Scores}) > \theta_{reid}$, issue **"ReID Confirmed"**, re-initialize Tracker on Winner candidate, and return to **State 1**.

---

## REQUEST FOR CLAUDE
Please generate a modular, clean, and production-ready **PyTorch Implementation Plan & Codebase Structure**, organized into the following sequential modules:
1. `dataset.py` & `data_miner.py`: Anchor-first strided sampling, $P \times K$ Sampler, and augmentation pipeline.
2. `models/`: VRU Backbone loading, STS Mamba Memory Module, and ReID Head.
3. `losses.py`: ID Loss, $L_2$-Normalized Hard-Triplet Loss, $L_2$-Normalized Center Loss, and Temporal Consistency Loss.
4. `train.py`: Two-stage learning rate schedule, Warm-up, Gradient Clipping, and Training Loop.
5. `evaluate.py`: Multi-object Query/Gallery Matrix Evaluation (Rank-1, mAP, mINP, and Latency Simulation).