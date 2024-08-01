# MICCAI 2023: FSDiffReg: Feature-wise and Score-wise Diffusion-guided Unsupervised Deformable Image Registration for Cardiac Images

This repository is the official implementation of "FSDiffReg: Feature-wise and Score-wise Diffusion-guided Unsupervised Deformable Image Registration for Cardiac Images".
<img src="./img/mainfigure.png">

## Requirements
Please use command
```
pip install -r requirements.txt
```
to install the environment. We used PyTorch 1.12.0, Python 3.8.10 for training.

## Data
* We used 3D cardiac MR images for training: [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

## Training

To run training process:

```bash
python train.py -c config/train_3D.json

CUDA_VISIBLE_DEVICES=1 python train_nohup.py -c config/train_3D_ACDC.json

CUDA_VISIBLE_DEVICES=1 python train_ACDC_sample_nohup.py -c config/train_3D_ACDC.json

nohup bash train_ACDC_nohup.sh > /home/liuhongzhi/Data/FSDiffReg_checkpoints/0721_v1.txt 2>&1 &

nohup bash train_ACDC_nohup.sh > /home/liuhongzhi/Data/FSDiffReg_checkpoints/0721_v2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train_nohup.py -c config/train_3D_ACDC.json

CUDA_VISIBLE_DEVICES=1 python train_ACDC_sample.py -c config/trainval_3D_ACDC.json

nohup bash train_ACDC_nohup.sh > /home/liuhongzhi/Data/FSDiffReg_checkpoints/20240730_val_freq_1.txt 2>&1 &

nohup bash train_ACDC_nohup.sh > /home/liuhongzhi/Data/FSDiffReg_checkpoints/20240730_val_freq_5.txt 2>&1 &

nohup bash train_ACDC_nohup.sh > /home/liuhongzhi/Data/Image_Registration/FSDiffReg_results/20240730_val_freq_1_v2.txt 2>&1 &

nohup bash train_ACDC_nohup.sh > /home/liuhongzhi/Data/Image_Registration/FSDiffReg_results/20240730_val_freq_5_v2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python DM_train_ACDC_sample.py -c config/diffuseMorph_trainval_3D_ACDC.json

nohup bash DM_train_ACDC_nohup.sh > /home/liuhongzhi/Data/Image_Registration/FSDiffReg_results/DiffuseMorph_20240730_val_freq_1_v1.txt 2>&1 &
```

## Test

To run testing process:

```
python3 test.py -c config/test_3D.json -w [YOUR TRAINED WEIGHTS]
```
Trained model can be found [here](https://drive.google.com/drive/folders/1x4NC9hHor2JexrclDmUMfKYTHhOQvVYT?usp=sharing)

## Acknowledgement

We would like to thank the great work of the following open-source project: [DiffuseMorph](https://github.com/DiffuseMorph/DiffuseMorph).

## Citation

