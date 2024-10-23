# Recursive Decomposition Network for Deformable Image Registration

**[Recursive Decomposition Network for Deformable Image Registration](https://ieeexplore.ieee.org/document/9826364)**

This repository is the implementation of the above paper, linked to [VIDAR Lab](https://vidar-ustc.github.io/biomedical-imaging/registration)


## Requirements
The packages and their corresponding version we used in this repository are listed below.
- Python 3
- Pytorch 1.1
- Numpy
- SimpleITK

## Training

After configuring the environment, please use this command to train the model.
```python
python -m torch.distributed.launch --nproc_per_node=4 train.py --epoch=xx --dataset=brain  --data_path=/xx/xx/  --base_path=/xx/xx/

```

```python

CUDA_VISIBLE_DEVICES=2 python train_OASIS.py

```

## Testing
Use this command to obtain the testing results.
```python
python eval.py  --dataset=brain --dataset_val=xx --restore_ckpt=xx --local_rank=0 --data_path=/xx/xx/  --base_path=/xx/xx/
```


