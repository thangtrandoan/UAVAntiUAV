# Kiến Trúc UAVReIDNet — Mermaid Diagrams

## 1. Tổng Quan End-to-End (UAVReIDNet)

```mermaid
flowchart TD
    classDef input fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000
    classDef backbone fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef temporal fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef head fill:#c8e6c9,stroke:#388e3c,stroke-width:1px,color:#000
    classDef tensor fill:#fff9c4,stroke:#fbc02d,stroke-width:1px,color:#000
    classDef op fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef result fill:#b2dfdb,stroke:#00796b,stroke-width:2px,color:#000

    Input(["📹 Video Clip<br>[B, N, 3, 224, 224]<br>B=batch, N=16 frames"]):::input

    Fold["Fold: gộp B×N<br>[B×N, 3, 224, 224]"]:::op

    subgraph GASNET ["🔬 1. GASNet — Visual Backbone (2D CNN)"]
        direction TB
        Stem["Stem: Conv7×7 → BN → ReLU → MaxPool<br>[B×N, 64, 56, 56]"]:::backbone
        L1["Layer 1 (IBN-a): ResBlock ×3<br>[B×N, 256, 56, 56]"]:::backbone
        L2["Layer 2 (IBN-a): ResBlock ×4<br>[B×N, 512, 28, 28]"]:::backbone
        L3["Layer 3 (IBN-a): ResBlock ×6<br>[B×N, 1024, 14, 14]"]:::backbone
        L4["Layer 4 (BN only): ResBlock ×3<br>[B×N, 2048, 7, 7]"]:::backbone

        RGA_S["RGA Spatial Attention<br>Relation Matrix 49×49"]:::backbone
        RGA_C["RGA Channel Attention<br>Relation Matrix 2048×2048"]:::backbone

        FS["OSBlockFS ×2 (fs1 → fs2)<br>Mỗi block có 4 streams Lite3×3 + Channel Gate<br>[B×N, 512, 14, 14]"]:::backbone

        GeM_G["GeM Pooling (p=3)<br>[B×N, 2048]"]:::backbone
        GeM_FS["GeM Pooling (p=3)<br>[B×N, 512]"]:::backbone

        BNNeck_G["BNNeck_Global (bias=0)<br>[B×N, 2048]"]:::backbone
        BNNeck_FS_bb["BNNeck_FS (bias=0)<br>[B×N, 512]"]:::backbone

        Stem --> L1 --> L2 --> L3
        L3 --> L4
        L4 --> RGA_S --> RGA_C
        L3 --> FS

        RGA_C --> GeM_G --> BNNeck_G
        FS --> GeM_FS --> BNNeck_FS_bb
    end

    ConcatV["Concat (Global + FS)<br>[B×N, 2560]"]:::op
    Reshape["Reshape: view(B, N, -1)<br>[B, N, 2560]"]:::op
    MeanPool_V["Temporal Mean Pooling<br>visual_feat [B, 2560]"]:::op

    subgraph MAMBA ["🧠 2. Temporal Mamba Encoder (SSM)"]
        direction TB
        LinProj["Linear Projection<br>2560 → 512"]:::temporal
        PosEmb["+ Learnable Positional Embedding<br>[B, N, 512]"]:::temporal

        subgraph BiMamba1 ["Bi-Mamba Layer 1 (1 S6 Block gọi 2 lần)"]
            direction TB
            Res1["res = x (lưu residual)"]:::op
            Fwd1["x_fwd = S6Block(x)<br>Chiều xuôi"]:::temporal
            FlipIn1["x.flip → S6Block → .flip<br>x_bwd: Chiều ngược"]:::temporal
            Sum1["x = LayerNorm(x_fwd + x_bwd + res)"]:::op
            Res1 --> Sum1
            Res1 --> Fwd1 & FlipIn1
            Fwd1 & FlipIn1 --> Sum1
        end

        subgraph BiMamba2 ["Bi-Mamba Layer 2 (1 S6 Block gọi 2 lần)"]
            direction TB
            Res2["res = x (lưu residual)"]:::op
            Fwd2["x_fwd = S6Block(x)<br>Chiều xuôi"]:::temporal
            FlipIn2["x.flip → S6Block → .flip<br>x_bwd: Chiều ngược"]:::temporal
            Sum2["x = LayerNorm(x_fwd + x_bwd + res)"]:::op
            Res2 --> Sum2
            Res2 --> Fwd2 & FlipIn2
            Fwd2 & FlipIn2 --> Sum2
        end

        TSeq["temporal_seq [B, N, 512]<br>(cho Temporal Consistency Loss)"]:::temporal
        MeanPool_T["Temporal Mean Pooling<br>[B, 512]"]:::temporal
        MLP["MLP Head: Linear → BN1d → ReLU<br>temporal_token [B, 512]"]:::temporal

        LinProj --> PosEmb
        PosEmb --> BiMamba1
        Sum1 --> BiMamba2
        Sum2 --> TSeq
        Sum2 --> MeanPool_T --> MLP
    end

    subgraph HEAD ["🎯 3. ReID Head"]
        direction TB
        ConcatH["Concat Visual + Temporal<br>[B, 2560 + 512 = 3072]"]:::head
        BNNeck["BNNeck (BatchNorm1d, bias=0)<br>[B, 3072]"]:::head
        Routing{"Train / Eval?"}:::op
        Classifier["Linear Classifier<br>[B, num_IDs]"]:::head
        BN_Feat_Out(["bn_feat [B, 3072]<br>→ Cosine Matching"]):::result

        ConcatH --> BNNeck --> Routing
        Routing -- "Train" --> Classifier
        Routing -- "Eval" --> BN_Feat_Out
    end

    Input --> Fold --> GASNET
    BNNeck_G -.->|"hoặc Raw GeM_G"| ConcatV
    BNNeck_FS_bb -.->|"hoặc Raw GeM_FS"| ConcatV
    ConcatV --> Reshape
    Reshape --> MeanPool_V
    Reshape --> MAMBA

    MeanPool_V --> ConcatH
    MLP --> ConcatH

    Classifier --> TrainOut(["(feat, bn_feat, logits)<br>→ Triplet Loss + ID Loss"]):::result
```

---

## 2. Chi Tiết: SimpleS6 Block (Selective State Space)

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef param fill:#fff9c4,stroke:#fbc02d,stroke-width:1px,color:#000

    X(["Input x [B, L, 512]"]):::tensor
    InProj["Linear in_proj<br>512 → 2048"]:::op
    XZ(["xz [B, L, 2048]"]):::tensor
    Chunk["Chunk chia đôi"]:::split

    X_Proj(["x_proj [B, L, 1024]"]):::tensor
    Z_Gate(["z_gate [B, L, 1024]"]):::tensor

    Conv1D["Depthwise Conv1d (k=4)<br>+ SiLU activation"]:::op
    X_Conv(["x_conv [B, L, 1024]"]):::tensor

    XProj2["Linear x_proj<br>1024 → 33"]:::op
    XDbl(["x_dbl [B, L, 33]"]):::tensor
    SplitSSM["Split: 1 + 16 + 16"]:::split

    DT_raw(["dt [B, L, 1]"]):::tensor
    B_mat(["B [B, L, 16]"]):::tensor
    C_mat(["C [B, L, 16]"]):::tensor

    DTProj["Linear + Softplus<br>1 → 1024"]:::op
    DT(["Δt [B, L, 1024]"]):::tensor

    A_Log(["A_log [1024, 16]<br>(Learnable)"]):::param
    A_Exp["-exp(A_log)"]:::op
    A_mat(["A [1024, 16]"]):::tensor

    subgraph PScan ["Parallel Selective Scan"]
        direction TB
        W_calc["W = Δt ⊗ A<br>[B, L, 1024, 16]"]:::op
        V_calc["V = (Δt ⊗ B) ⊗ x_conv<br>[B, L, 1024, 16]"]:::op
        CumSum["P = CumSum(W, dim=L)"]:::op
        Diff["M = P_i − P_j (Causal Mask)"]:::op
        ExpW["weights = exp(M) · tril_mask"]:::op
        H_calc["h = Σⱼ weights·V"]:::op
        H(["h [B, L, 1024, 16]"]):::tensor

        W_calc --> CumSum --> Diff --> ExpW --> H_calc --> H
        V_calc --> H_calc
    end

    Y_sum["y = Σ(h · C, dim=d_state)"]:::op
    Y(["y [B, L, 1024]"]):::tensor
    SkipD["y += x_conv · D"]:::op
    Gate["y = y · SiLU(z_gate)"]:::op
    OutProj["Linear out_proj<br>1024 → 512"]:::op
    Out(["Output [B, L, 512]"]):::tensor

    X --> InProj --> XZ --> Chunk
    Chunk --> X_Proj & Z_Gate
    X_Proj --> Conv1D --> X_Conv
    X_Conv --> XProj2 --> XDbl --> SplitSSM
    SplitSSM --> DT_raw & B_mat & C_mat
    DT_raw --> DTProj --> DT
    A_Log --> A_Exp --> A_mat

    DT --> W_calc & V_calc
    A_mat --> W_calc
    B_mat --> V_calc
    X_Conv --> V_calc

    H --> Y_sum
    C_mat --> Y_sum
    Y_sum --> Y --> SkipD
    X_Conv --> SkipD
    SkipD --> Gate
    Z_Gate --> Gate
    Gate --> OutProj --> Out
```

---

## 3. Chi Tiết: RGA (Relation-Aware Global Attention)

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000

    X(["Input [B×N, 2048, 7, 7]<br>N_nodes = 49"]):::tensor

    Theta["θ = Conv 1×1"]:::op
    Phi["φ = Conv 1×1"]:::op

    T_Out(["θ(X) [B×N, 49, C']"]):::tensor
    P_Out(["φ(X) [B×N, C', 49]"]):::tensor

    BMM["Batch MatMul<br>θ × φᵀ"]:::op
    R_Mat(["Relation Matrix<br>[B×N, 49, 49]"]):::tensor

    Concat_RowCol["Concat hàng & cột<br>[B×N, 98, 49]"]:::split
    Conv_Rel["Conv1d → Compress"]:::op
    Rel_Feat(["Relation Feature<br>[B×N, C_rel, 49]"]):::tensor

    G_Pool["Global Avg Pool<br>theo channel"]:::op
    G_Out(["Global Feature<br>[B×N, 1, 49]"]):::tensor

    Concat_All["Concat"]:::split
    Attn["Conv1d → Sigmoid"]:::op
    Mask(["Spatial Mask<br>[B×N, 1, 7, 7]"]):::tensor

    Mul["X ⊗ Mask"]:::op
    Out(["Output [B×N, 2048, 7, 7]"]):::tensor

    X --> Theta --> T_Out
    X --> Phi --> P_Out
    T_Out & P_Out --> BMM --> R_Mat
    R_Mat --> Concat_RowCol --> Conv_Rel --> Rel_Feat
    X --> G_Pool --> G_Out
    Rel_Feat & G_Out --> Concat_All --> Attn --> Mask
    X & Mask --> Mul --> Out
```

---

## 4. Chi Tiết: Full Scale Branch (2× OSBlockFS: fs1 → fs2)

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef block fill:#c8e6c9,stroke:#388e3c,stroke-width:1px,color:#000

    X(["Layer 3 / Stage 3 Output<br>[B×N, 1024, 14, 14]"]):::tensor

    subgraph FS1 ["Block FS 1: self.fs1 (OSBlockFS: 1024 → 512)"]
        direction TB
        Split1["Chia 4 Streams"]:::split
        S1_1["Stream 1: 1× Lite3×3 (Tầm gần)"]:::block
        S1_2["Stream 2: 2× Lite3×3 (Tầm vừa)"]:::block
        S1_3["Stream 3: 3× Lite3×3 (Tầm xa)"]:::block
        S1_4["Stream 4: 4× Lite3×3 (Siêu xa)"]:::block
        Gate1["Shared ChannelGate<br>→ 4 Masks [B×N, 512, 1, 1]"]:::op
        Sum1["Weighted Sum 4 Streams"]:::split
        Shortcut1["Shortcut: Conv 1×1 (1024 → 512)"]:::op
        AddReLU1["Add + ReLU"]:::op

        Split1 --> S1_1 & S1_2 & S1_3 & S1_4
        S1_1 & S1_2 & S1_3 & S1_4 --> Gate1
        S1_1 & S1_2 & S1_3 & S1_4 & Gate1 --> Sum1
        Sum1 & Shortcut1 --> AddReLU1
    end

    Mid(["fs1 Output<br>[B×N, 512, 14, 14]"]):::tensor

    subgraph FS2 ["Block FS 2: self.fs2 (OSBlockFS: 512 → 512)"]
        direction TB
        Split2["Chia 4 Streams"]:::split
        S2_1["Stream 1: 1× Lite3×3"]:::block
        S2_2["Stream 2: 2× Lite3×3"]:::block
        S2_3["Stream 3: 3× Lite3×3"]:::block
        S2_4["Stream 4: 4× Lite3×3"]:::block
        Gate2["Shared ChannelGate<br>→ 4 Masks [B×N, 512, 1, 1]"]:::op
        Sum2["Weighted Sum 4 Streams"]:::split
        Shortcut2["Shortcut: Identity (512 = 512)"]:::op
        AddReLU2["Add + ReLU"]:::op

        Split2 --> S2_1 & S2_2 & S2_3 & S2_4
        S2_1 & S2_2 & S2_3 & S2_4 --> Gate2
        S2_1 & S2_2 & S2_3 & S2_4 & Gate2 --> Sum2
        Sum2 & Shortcut2 --> AddReLU2
    end

    GeM["GeM Pooling (p=3)"]:::op
    FS_Feat(["Raw Feature: fs_feat<br>[B×N, 512]"]):::tensor
    BNNeck_FS["BNNeck_fs (BatchNorm1d, bias=0)<br>[B×N, 512]"]:::op
    BN_FS(["bn_fs<br>[B×N, 512]"]):::tensor
    Cls_FS["Linear Classifier (fs)<br>[B×N, num_classes]"]:::block

    X --> Split1
    X --> Shortcut1
    AddReLU1 --> Mid
    Mid --> Split2
    Mid --> Shortcut2
    AddReLU2 --> GeM --> FS_Feat
    FS_Feat --> BNNeck_FS --> BN_FS --> Cls_FS
    FS_Feat -.->|"Khi tích hợp UAVReIDNet<br>(return_raw_features_eval=True)"| ToReIDNet(["Concat với Global Feat<br>→ Đưa vào ReID Head"]):::tensor
```

---

## 5. Training Flow (Siamese Mode)

```mermaid
flowchart LR
    classDef input fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000
    classDef proc fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef loss fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef result fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000

    Before(["before_clips<br>[B, N, 3, H, W]"]):::input
    After(["after_clips<br>[B, N, 3, H, W]"]):::input

    Ext1["extract_features()"]:::proc
    Ext2["extract_features()"]:::proc

    Head1["ReID Head"]:::proc
    Head2["ReID Head"]:::proc

    Feat_B(["feat_b, bn_b, logits_b"]):::result
    Feat_A(["feat_a, bn_a, logits_a"]):::result
    T_Seq(["t_seq_before, t_seq_after"]):::result

    ID_Loss["Cross-Entropy<br>(ID Loss)"]:::loss
    Triplet_Loss["Triplet Loss<br>(Hard Mining)"]:::loss
    Temporal_Loss["Temporal<br>Consistency Loss"]:::loss

    Total["Total Loss<br>= λ₁·ID + λ₂·Triplet + λ₃·Temporal"]:::loss

    Before --> Ext1 --> Head1 --> Feat_B
    After --> Ext2 --> Head2 --> Feat_A
    Ext1 & Ext2 --> T_Seq

    Feat_B & Feat_A --> ID_Loss & Triplet_Loss
    T_Seq --> Temporal_Loss
    ID_Loss & Triplet_Loss & Temporal_Loss --> Total
```

---

## 6. Inference Flow

```mermaid
flowchart LR
    classDef input fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000
    classDef proc fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef result fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000

    Clip(["Video Clip<br>[1, N, 3, 224, 224]"]):::input
    Backbone["GASNet Backbone<br>(frozen, eval mode)"]:::proc
    Temporal["Temporal Mamba<br>Encoder"]:::proc
    Head["ReID Head<br>(eval: return bn_feat)"]:::proc
    Feature(["Identity Feature<br>bn_feat [1, 3072]"]):::result
    Match["Cosine Similarity<br>Distance Matrix"]:::proc
    Result(["Re-ID Result<br>Rank-1 / mAP"]):::result

    Clip --> Backbone --> Temporal --> Head --> Feature --> Match --> Result
```
