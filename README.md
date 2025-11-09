# Hierarchical Medical Image Classification with MedMNIST3D

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **three-stage hierarchical coarse-to-fine learning approach** for robust 3D medical image classification across multiple anatomical regions using the MedMNIST3D dataset.

## Research Question

**How can a hierarchical, oncology-guided coarse-to-fine learning model improve the robustness of medical image classification across multiple anatomical regions such as brain, abdomen, and chest?**

## Overview

This research implements a hierarchical classification pipeline that mimics radiological diagnostic reasoning:

```
Medical Image → [Stage 1] Region Localization → [Stage 2] Pathology Detection → [Stage 3] Subtype Classification
                (brain/abdomen/chest)          (region-specific diseases)      (disease subtypes)
```

### Three-Stage Pipeline

1. **Stage 1 (Coarse)**: Anatomical Region Localization
   - Classifies images into anatomical regions: brain, abdomen, chest
   - Uses 3D CNNs to learn region-specific features
   - Mimics initial localization in radiological workflow

2. **Stage 2 (Fine)**: Region-Specific Pathology Classification
   - Separate specialized models for each anatomical region
   - Focused disease detection within identified regions
   - Reduces inter-region pathology confusion

3. **Stage 3 (Subtype)**: Disease Subtype Identification (Optional)
   - Fine-grained disease categorization
   - Detailed pathology subtype classification
   - Enhanced diagnostic specificity