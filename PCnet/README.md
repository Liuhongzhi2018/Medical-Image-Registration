#  Joint Progressive and Coarse-to-Fine Registration of Brain MRI via Deformation Field Integration and Non-Rigid Feature Fusion (PCnet)

# training code
```bash

CUDA_VISIBLE_DEVICES=2 python train_LPBA.py

CUDA_VISIBLE_DEVICES=0 python train_ACDC.py

CUDA_VISIBLE_DEVICES=3 python train_OASIS.py

CUDA_VISIBLE_DEVICES=3 python train_OAIZIB.py

```