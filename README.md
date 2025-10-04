# Vision Transformer (ViT) on CIFAR-10

This repository implements a **Vision Transformer (ViT)** for image classification on the CIFAR-10 dataset using PyTorch. The model follows the architecture from the paper:

**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
Dosovitskiy et al., ICLR 2021.

---

## 🚀 Goal

Train a ViT to classify CIFAR-10 images (10 classes) with high accuracy. The solution includes **data augmentations, MixUp, learning rate scheduler, and EMA** for improved performance.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch 2.x
- Torchvision
- CUDA-enabled GPU recommended
- Run exclusively on **Google Colab**

---

## ⚡ How to Run in Colab

Run all the cells one by one.

## 🧩 Best Model Configuration
Parameter	Value
Image Size	32x32
Patch Size	4x4
Embedding Dim	256
Depth (Transformer)	6
Number of Heads	4
MLP Ratio	4
Dropout	0.1
Drop Path	0.1
Optimizer	AdamW
Learning Rate	3e-4
Weight Decay	0.05
Scheduler	Cosine with 5% warmup
Batch Size	128
Epochs	200
MixUp	alpha=0.8
Label Smoothing	0.1

## 📝 Results (CIFAR-10)
Metric	Value (%)
Training Acc	~49.35
Validation Acc	~89.24

## 🔬 Optional Analysis / Notes
Patch size vs accuracy: Smaller patches (4x4) preserve more local details, boosting accuracy.

Depth vs width: Moderate depth (6 blocks) with medium embedding (256) provides good trade-off between performance and training time.

Augmentations: RandomCrop + RandAugment + MixUp significantly improve generalization.

Optimizer & Scheduler: AdamW with cosine decay and warmup stabilizes training.

EMA: Helps in smoothing weights, slightly improving validation accuracy.

## 📖 References
ViT Paper (ICLR 2021)

PyTorch Documentation

CIFAR-10 Dataset


# Text-Prompted Segmentation with SAM 2 (Contour-Based)
## Overview

This project implements text-prompted segmentation of a single image using Segment Anything Model (SAM 2).
Instead of using CLIP or GroundingDINO for text-to-region mapping, a simple heuristic is used based on color and size.

Candidate object regions are generated using contour detection in OpenCV. The region most likely matching the text prompt is then fed into SAM 2 for precise mask prediction.

## Features

Runs fully on Google Colab GPU.

Uses SAM 2 for accurate segmentation masks.

Simple text-to-region mapping without external vision-language models.

Works end-to-end with Drive-hosted checkpoint and image.

Visualizes the final mask overlay on the original image.

## Requirements

Python 3.8+

PyTorch 2.x

OpenCV 4.7+ (with contrib)

segment-anything (SAM 2)

PIL, matplotlib, numpy

## How to Run in Colab

Clone or copy the notebook (q2.ipynb) into Colab.

Mount Google Drive and place:

SAM checkpoint: MyDrive/checkpoints/sam_vit_b_01ec64.pth

Image to segment: MyDrive/resources/sample.jpg

Run all cells in order.

Modify the text prompt.

The notebook outputs the segmentation mask overlay on the image.

## Pipeline

Load image from Google Drive.

Generate candidate regions using OpenCV contours.

Text prompt → region heuristic:

If prompt contains "red", "green", "blue", choose region with dominant color.

Otherwise, fallback to largest region.

Feed selected region to SAM 2 → predict precise mask.

Visualize mask overlay on the original image.

## Limitations

The text-to-region mapping is heuristic-based, so it works best for objects with distinctive colors or largest size.

Cannot handle complex queries like "small white cat under tree" accurately.

Only supports single image segmentation (no multi-object detection yet).

For more accurate text-guided segmentation, integration with vision-language models like CLIP, GroundingDINO, or GLIP is recommended.

## Citation

Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021 (for SAM reference).

Author: Jai Kumar
Date: 2025-10-04
