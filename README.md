# Parameter-Efficient Training-Free Domain Adaptation via LoRA Fusion for Open-Vocabulary Semantic Segmentation

This repository contains the implementation of **parameter-efficient, training-free domain adaptation for Open-Vocabulary Semantic Segmentation (OVSS)** using domain-specific **LoRA/Block-LoRA adapters** and **centroid-guided adapter retrieval and fusion**.

The proposed framework aims to improve the robustness of CLIP-based semantic segmentation under domain shifts without requiring target-domain retraining or gradient-based optimization during inference.

## Overview

Open-Vocabulary Semantic Segmentation enables semantic segmentation beyond a predefined set of training classes by leveraging the knowledge of vision-language models. However, its performance can degrade significantly when the target domain differs from the training distribution, particularly under heterogeneous environmental conditions such as fog, rain, snow, and nighttime scenes.

To address this problem, this work constructs a library of domain-specific LoRA/Block-LoRA adapters during an offline stage. During inference, the target input is represented in a shared embedding space and compared with domain centroids. The most relevant adapters are then retrieved and fused according to their similarity to the target representation.

The framework therefore separates **domain-specific knowledge acquisition** from **knowledge utilization during inference**.

### Key Features

* CLIP-based Open-Vocabulary Semantic Segmentation
* Parameter-efficient LoRA and Block-LoRA adaptation
* Library of LoRA-based Adapter 
* Centroid-guided adapter retrieval
* Similarity-guided adapter fusion
* Gradient-Free test-time adaptation
* Frozen CLIP backbone during adaptation
* Support for heterogeneous domain shifts

## Framework

The proposed framework consists of three main stages:

1. **Base OVSS Model**
   A CLIP-based open-vocabulary semantic segmentation model provides the shared visual-semantic representation.

2. **Domain-Specific Adapter Library**
   LoRA or Block-LoRA adapters are trained offline on different source domains and stored in an adapter library. A domain centroid is constructed to represent the visual characteristics of each source domain.

3. **Training-Free Test-Time Adaptation**
   During inference, the target representation is compared with the stored domain centroids. Relevant adapters are retrieved and fused according to their similarity, producing a target-aware adapted model without gradient-based optimization.

```text
                    Offline Stage
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   Source Domains                    CLIP-based OVSS
        │                                 │
        ▼                                 │
 LoRA / Block-LoRA                        │
    Training                              │
        │                                 │
        ▼                                 │
 Domain-Specific Adapter Library          │
        │                                 │
        ▼                                 │
   Domain Centroids                       │
        │                                 │
        └──────────────┬──────────────────┘
                       │
                       ▼
                 Test-Time Stage
                       │
                Target Input
                       │
                       ▼
             Target Representation
                       │
                       ▼
          Centroid-Guided Retrieval
                       │
                       ▼
          Similarity-Guided Fusion
                       │
                       ▼
            Target-Specific Adapter
                       │
                       ▼
             Segmentation Output
```

## Experimental Setup

### Backbone

* Vision-Language Model: **CLIP**
* Vision Encoder: **ViT-L/14**
* Framework: **PyTorch**
* GPU: **NVIDIA GeForce RTX 2080**
* Input Resolution: **512 × 512**

### Adaptation Methods

The repository supports:

* Vanilla LoRA
* Block-LoRA
* Uniform Adapter Fusion
* Similarity-Guided Adapter Retrieval and Fusion

### Datasets

The experiments use multiple datasets to evaluate cross-domain generalization and adaptation performance, including:

* IDD
* PASCAL Context 59 (PC59)
* NYU Depth V2
* ACDC
* MUSES
* ADE20K
* BDD100K
* Cityscapes
* Mapillary Vistas

The datasets are organized into source and target domains according to the experimental protocol.

## Installation

Clone the repository:

```bash
git clone https://github.com/Cheer3142/Gradient-FREE-Domain-Adaptation/blob/main
```

Create the Python environment:

```bash
conda create -n ovss-lora python=3.10
conda activate ovss-lora
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the required datasets from their official sources and organize them according to the expected directory structure.

```text
datasets/
├── IDD/
├── PC59/
├── NYUv2/
├── ACDC/
├── MUSES/
├── ADE20K/
├── BDD100K/
├── Cityscapes/
└── Mapillary/
```

Dataset paths should be updated in the corresponding configuration files before training or evaluation.

## Training Domain-Specific Adapters

To train a LoRA adapter for a specific source domain:

```bash
python train_lora.py \
    --dataset <DATASET_NAME> \
    --rank 16 \
    --alpha 16 \
    --lr 1e-4
```

For Block-LoRA:

```bash
python train_block_lora.py \
    --dataset <DATASET_NAME> \
    --rank 16 \
    --alpha 16 \
    --lr 1e-4
```

The trained adapters are stored in the adapter library:

```text
adapters/
├── domain_1/
├── domain_2/
├── domain_3/
└── ...
```

The inference pipeline performs:

1. Target image feature extraction
2. Domain similarity calculation
3. Centroid-guided adapter retrieval
4. Similarity-guided adapter fusion
5. Open-vocabulary semantic segmentation

## Results

The proposed framework improves the robustness of OVSS under heterogeneous domain shifts while avoiding target-domain retraining.

| Method                             | Average mIoU |
| ---------------------------------- | -----------: |
| Zero-Shot OVSS                     |        32.13 |
| LoRA Adaptation                    |        50.75 |
| Block-LoRA Adaptation              |        50.78 |
| Uniform Fusion (LoRA)              |        41.10 |
| Uniform Fusion (Block-LoRA)        |        38.41 |
| Similarity-Guided TTA (LoRA)       |        43.55 |
| Similarity-Guided TTA (Block-LoRA) |    **43.91** |

The results demonstrate that domain-specific adapters can substantially improve segmentation performance compared with the zero-shot baseline. The proposed similarity-guided mechanism further enables training-free adaptation by dynamically selecting and combining relevant domain-specific knowledge.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@thesis{sombatsatien2026parameter,
  title     = {Parameter-Efficient Training-Free Domain Adaptation via LoRA Fusion for Open-Vocabulary Semantic Segmentation},
  author    = {Sombatsatien, Patcharadanai},
  year      = {2026},
  school    = {Xi'an Jiaotong University}
}
```

## Acknowledgements

This work was conducted as part of a research project on Open-Vocabulary Semantic Segmentation, parameter-efficient adaptation, and vision-language models.

