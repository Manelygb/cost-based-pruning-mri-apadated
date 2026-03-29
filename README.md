# Pruning + Finetuning pipeline for MRI unet

This repository contains a refactored pruning pipeline (Taylor criterion → channel selection → structured channel pruning → finetune → evaluation) for a U-Net trained on the Brain MRI segmentation dataset.

This is not the updated version, the code is still not organized, it was edited just to prove results already proven on a nother dataset : Carvana Segmentation dataset.
## Reference
### Dataset
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
### U-Net PyTorch Implementation
https://github.com/milesial/Pytorch-UNet
### U-Net Paper
https://arxiv.org/pdf/1505.04597.pdf
### Pruning Paper
https://arxiv.org/abs/1611.06440
