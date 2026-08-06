# PROBLEM STATEMENT & TECHNICAL REQUIREMENTS FOR CLAUDE CODE PLAN

## 1. Context & Task Definition
- **Project Name:** Air-to-Air UAV Re-Identification under Out-of-View & Re-appearance Scenarios.
- **Problem Statement:** In an aerial pursuit scenario (Air-to-Air UAV Tracking), a pursuer UAV tracks a target UAV. When the target UAV disappears from the camera's Field-of-View (Out-of-View / Full Occlusion) and reappears after a short/long period under severe viewpoint/scale changes and motion blur, the system must accurately **Re-Identify (ReID)** the original target without false-matching other distractor UAVs or environmental noise.
- **Core Paradigm:** Video-based Cross-Temporal Person/Object ReID & Multi-Modal Target Verification (Visual + Temporal Memory + Language Prompts).

---

## 2. Model Architecture Specifications

### A. Visual Backbone (Feature Extractor)
- **Architecture:** Hierarchical Vision Transformer (HiViT-Base or Swin-Transformer Base).
- **Role:** Extracts fine-grained multi-scale spatial representations (body/rotor structure, color/frame patterns) robust to tiny scales ($<1\%$ image area) and motion blur.

### B. Temporal Memory Engine
- **Architecture:** Selective State Space Model (Vision Mamba / Vim block with 1D Unidirectional Scanning).
- **Role:** Maintains a continuous latent state (**Temporal Token / Memory Bank**) from $N$ contiguous frames before target disappearance ($T_{before}$). Compresses historical trajectory and appearance cues with $\mathcal{O}(N)$ linear time complexity.

### C. Optional Semantic Anchor (Vision-Language Alignment)
- **Architecture:** Pre-trained Text Encoder (e.g., Mamba-130M / CLIP Text Encoder).
- **Role:** Encodes natural language descriptions (Language Prompts describing target category, color, behavior) to act as a semantic anchor during re-identification.

---

## 3. Loss Functions & Joint Optimization

The overall loss is a multi-task joint optimization function:
$$\mathcal{L}_{total} = \mathcal{L}_{ID} + \lambda_1 \mathcal{L}_{Hard-Triplet} + \lambda_2 \mathcal{L}_{Temporal\_Consistency} + \lambda_3 \mathcal{L}_{Center}$$

1. **$\mathcal{L}_{ID}$ (Label-Smoothed Cross-Entropy Loss):** Ensures global class discriminability.
2. **$\mathcal{L}_{Hard-Triplet}$ (Hard-Margin Triplet Loss):** Applied on $L_2$-normalized feature vectors. Enforces matching between $T_{before}$ (Anchor) and $T_{after}$ (Positive) while pushing Distractor UAVs (Negative) beyond a predefined margin.
3. **$\mathcal{L}_{Center}$ (Center Loss):** Pulls all variations ($T_{before}, T_{after}$, occluded states) toward the centroid of the target ID ($||f_i - c_y||_2^2$). Applied on $L_2$-normalized vectors with $\lambda_3 \approx 0.001 - 0.05$.
4. **$\mathcal{L}_{Temporal\_Consistency}$:** Enforces smooth latent state transitions in the Mamba Memory Module between adjacent frames to prevent feature drift at occlusion boundaries.

---

## 4. Data Engineering Strategy

### Phase 1: Pre-training (Visual Backbone Only)
- **Data Source:** Large-scale Aerial/Vehicle ReID datasets (`AG-ReID.v2`, `VRU`, `VRAI`).
- **Objective:** Learn general high-altitude/top-down visual feature representations.

### Phase 2: Fine-Tuning (Full ReID + Mamba Memory Engine)
- **Data Source:** `UAV-Anti-UAV Dataset (2025)` Train Split.
- **Pairing Strategy:**
  - Extract frame pairs/clips of the same target UAV **before disappearance ($T_{before}$)** and **after re-appearance ($T_{after}$)** from sequences labeled with `OV` (Out-of-View) or `FO` (Full Occlusion).
  - Inject Hard Negative Distractor UAV crops from `Anti-UAV318` / `Anti-UAV600` / `DUT Anti-UAV` into the training batches.
- **Augmentations:** Random Erasing/Cutout (simulating partial occlusion), Motion Blur, Scale Resizing, Color Jitter.

---

## 5. Evaluation Protocol & Metrics

The system will be evaluated under 3 primary scenarios: (1) Short Disappearance ($<5s$), (2) Long Disappearance ($>15s$), and (3) Re-appearance with Distractors (multiple candidate UAVs in frame).

### A. Offline ReID Evaluation (Static Protocol)
- **Query Set:** Crop images/clips of the target UAV immediately after re-appearance ($T_{after}$).
- **Gallery Set:** Crop images/clips of the target before disappearance ($T_{before}$) + $K$ Distractor UAVs & environmental noise (fixed size, e.g., $K = 1,000$).
- **Metrics:**
  - **Rank-1 Accuracy (%)**
  - **mAP (mean Average Precision)**
  - **mINP (mean Inverse Negative Penalty):** Evaluates hard-sample retrieval quality.

### B. Online Sequential Evaluation (Stream Protocol)
- **Protocol:** Simulate real-time video stream starting from $T_{after}$. Perform Object Detection $\rightarrow$ Batch Candidate ReID Scoring.
- **Metrics:**
  - **Re-acquisition Latency (Frames/ms):** Time taken to issue `Confirmed Match` ($Score > \theta_{reid}$).
  - **False Alarm Rate (%):** Rate of incorrectly locking onto a non-target UAV/distractor.

---

## 6. Real-Time Inference Logic (Target Pipeline)

1. **State 1: Tracking Mode ($Confidence > Threshold_{track}$)**
   - Tracker updates bounding box.
   - Mamba Memory Engine continuously ingests target crops and updates the latent `Temporal Token`.
2. **State 2: Lost Mode ($Confidence \le Threshold_{track}$)**
   - Target disappears (`OV` or `FO`).
   - Freeze the Mamba `Temporal Token` state (lock memory).
3. **State 3: ReID Verification Mode (New Candidates Detected)**
   - Detector identifies $M$ candidates $[C_1, C_2, ..., C_M]$ in frame.
   - Extract feature vectors in a **single GPU batch**: $[V_1, V_2, ..., V_M]$.
   - Compute Similarity Matrix against locked `Temporal Token` (+ Kinematic/Motion Filter).
   - If $\max(\text{Scores}) > \theta_{reid}$, issue **"ReID Confirmed"**, re-initialize Tracker on Winner candidate, and return to **State 1**.

---

## EXPECTED OUTPUT FROM CLAUDE
Based on the problem statement above, please provide a structured, modular, and production-ready **Code Implementation Plan**, including:
1. **Directory Structure & Data Pipeline Scripting** (Dataset pre-processing, Pair/Clip extraction for $T_{before} \leftrightarrow T_{after}$).
2. **Model Architecture Implementation** (HiViT Backbone + STS Mamba Memory Module + Head).
3. **Loss Functions & Training Loop Implementation** (Batch construction, Triplet/Center/ID Loss calculation).
4. **Evaluation Module Implementation** (Offline Rank-1/mAP/mINP calculation & Online Video Stream ReID verification benchmark).

## NOTE
Note for Phase 1: We already have a pre-trained ReID model checkpoint trained on the VRU dataset. We will bypass scratch pre-training and load these pre-trained weights directly into our Visual Backbone before Fine-Tuning.

UAV-Anti-UAV data path: /thang/UAV-Anti-UAV
model: in /thang/gasnet_project, use Resnet-50 IBN as backbone
weight path: /thang/gasnet_project/test/gasnet.best.pth