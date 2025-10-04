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

Author: Jai Kumar
Date: 2025-10-04
