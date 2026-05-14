# Exp 1: image-only ViT-UNetSeg for KiTS23

This patch implements USCNet-style image-only segmentation:

```text
3D CT image
-> Visual Transformation / 3D Patch Embedding
-> ViT Encoder
-> Z3 / Z6 / Z9 / Z12 features
-> UNETR-style decoder
-> 4-class KiTS23 segmentation
```

## Files

```text
models/ViT_UNetSeg3D.py
data/dataloader_exp1_vit_unetseg.py
segmentation_exp1_vit_unetseg.py
inference_exp1_vit_unetseg.py
main_exp1_vit_unetseg.py
requirements_exp1_vit_unetseg.txt
```

## Dataset structure

```text
dataset/kits23/labeled/
  train/
    images/*.npz
    labels/*.npz
  valid/
    images/*.npz
    labels/*.npz
  test/
    images/*.npz
    labels/*.npz
```

Each `.npz` file is expected to contain:

```python
data
```

Label mapping:

```text
0 = background
1 = kidney
2 = tumor
3 = cyst
```

## Install

```bash
pip install -r requirements_exp1_vit_unetseg.txt
```

## Run

```bash
python main_exp1_vit_unetseg.py
```

## Important settings

In `main_exp1_vit_unetseg.py`:

```python
ROI_SIZE = (96, 128, 128)
NORMALIZE = False
```

`ROI_SIZE` must be divisible by `16` because patch size is 16.

If OOM:

```python
ROI_SIZE = (64, 96, 96)
hidden_size = 192
mlp_dim = 768
```
