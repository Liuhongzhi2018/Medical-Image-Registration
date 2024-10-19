# Self-Distilled Hierarchical Network

**[Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration](https://ieeexplore.ieee.org/abstract/document/10042453)**

IEEE Transactions on Medical Imaging (TMI) 2023

Shenglong Zhou, Bo Hu, Zhiwei Xiong and Feng Wu

University of Science and Technology of China (USTC)

## Introduction

![Framework](https://user-images.githubusercontent.com/26156941/201927630-23340d83-52a0-45b6-a007-19c7fb603ea9.png)

We present a novel unsupervised learning approach named Self-Distilled Hierarchical Network (SDHNet).  
By decomposing the registration procedure into several iterations, SDHNet generates hierarchical deformation fields (HDFs) simultaneously in each iteration and connects different iterations utilizing the learned hidden state.
Hierarchical features are extracted to generate HDFs through several parallel GRUs, and HDFs are then fused adaptively conditioned on themselves as well as contextual features from the input image.
Furthermore, different from common unsupervised methods that only apply similarity loss and regularization loss, SDHNet introduces a novel self-deformation distillation scheme. 
This scheme distills the final deformation field as the teacher guidance, which adds constraints for intermediate deformation fields.

## Requirements
The packages and their corresponding version we used in this repository are listed below.
- Python 3
- Pytorch 1.1
- Numpy
- SimpleITK

## Training
After configuring the environment, please use this command to train the model.

```python
python -m torch.distributed.launch --nproc_per_node=4 train.py  --name=SDHNet  --iters=6 --dataset=brain  --data_path=/xx/xx/  --base_path=/xx/xx/

```

```bash

CUDA_VISIBLE_DEVICES=2 python train_ACDC.py

CUDA_VISIBLE_DEVICES=3 python train_LPBA.py

CUDA_VISIBLE_DEVICES=2 python train_OASIS.py

CUDA_VISIBLE_DEVICES=2 python train_OAIZIB.py

```

transpose image1 shape: torch.Size([1, 1, 80, 80, 80]) image2 shape: torch.Size([1, 1, 80, 80, 80])
_interpolate im: torch.Size([1, 1, 80, 80, 80])
_interpolate base: torch.Size([512000])
_interpolate idx_a: torch.Size([512000, 1])
self.affnet: torch.Size([1, 1, 80, 80, 80]) torch.Size([1, 1, 80, 80, 80])
AffineNet conv1 torch.Size([1, 16, 40, 40, 40])
AffineNet conv2 torch.Size([1, 32, 20, 20, 20])
AffineNet conv3_1 torch.Size([1, 64, 10, 10, 10])
AffineNet conv4_1 torch.Size([1, 128, 5, 5, 5])
AffineNet conv5_1 torch.Size([1, 256, 3, 3, 3])
AffineNet conv6_1 torch.Size([1, 512, 2, 2, 2])
AffineNet conv7_W Conv3d(512, 9, kernel_size=(2, 2, 2), stride=(1, 1, 1), bias=False) torch.Size([1, 512, 2, 2, 2]) 1
AffineNet W torch.Size([1, 3, 3]) b torch.Size([1, 3])
AffineNet flow torch.Size([1, 3, 80, 80, 80]) A torch.Size([1, 3, 3]) W torch.Size([1, 3, 3]) b torch.Size([1, 3])
reconstruction:  torch.Size([1, 1, 80, 80, 80]) torch.Size([1, 3, 80, 80, 80])
_interpolate im: torch.Size([1, 1, 80, 80, 80])
_interpolate base: torch.Size([512000])
_interpolate idx_a: torch.Size([512000, 1])


transpose image1 shape: torch.Size([1, 1, 160, 160, 160]) image2 shape: torch.Size([1, 1, 160, 160, 160])
_interpolate im: torch.Size([1, 1, 160, 160, 160])
_interpolate base: torch.Size([4096000])
_interpolate idx_a: torch.Size([4096000, 1])
self.affnet: torch.Size([1, 1, 160, 160, 160]) torch.Size([1, 1, 160, 160, 160])
AffineNet conv1 torch.Size([1, 16, 80, 80, 80])
AffineNet conv2 torch.Size([1, 32, 40, 40, 40])
AffineNet conv3_1 torch.Size([1, 64, 20, 20, 20])
AffineNet conv4_1 torch.Size([1, 128, 10, 10, 10])
AffineNet conv5_1 torch.Size([1, 256, 5, 5, 5])
AffineNet conv6_1 torch.Size([1, 512, 3, 3, 3])
AffineNet conv7_W Conv3d(512, 9, kernel_size=(2, 2, 2), stride=(1, 1, 1), bias=False) torch.Size([1, 512, 3, 3, 3]) 1
AffineNet W torch.Size([8, 3, 3]) b torch.Size([8, 3])
AffineNet flow torch.Size([8, 3, 160, 160, 160]) A torch.Size([8, 3, 3]) W torch.Size([8, 3, 3]) b torch.Size([8, 3])
reconstruction:  torch.Size([1, 1, 160, 160, 160]) torch.Size([8, 3, 160, 160, 160])
_interpolate im: torch.Size([1, 1, 160, 160, 160])
_interpolate base: torch.Size([4096000])

transpose image1 shape: torch.Size([1, 1, 128, 128, 128]) image2 shape: torch.Size([1, 1, 128, 128, 128])
_interpolate im: torch.Size([1, 1, 128, 128, 128])
_interpolate base: torch.Size([2097152])
_interpolate idx_a: torch.Size([2097152, 1])
self.affnet: torch.Size([1, 1, 128, 128, 128]) torch.Size([1, 1, 128, 128, 128])
AffineNet conv1 torch.Size([1, 16, 64, 64, 64])
AffineNet conv2 torch.Size([1, 32, 32, 32, 32])
AffineNet conv3_1 torch.Size([1, 64, 16, 16, 16])
AffineNet conv4_1 torch.Size([1, 128, 8, 8, 8])
AffineNet conv5_1 torch.Size([1, 256, 4, 4, 4])
AffineNet conv6_1 torch.Size([1, 512, 2, 2, 2])
AffineNet conv7_W Conv3d(512, 9, kernel_size=(2, 2, 2), stride=(1, 1, 1), bias=False) torch.Size([1, 512, 2, 2, 2]) 1
AffineNet W torch.Size([1, 3, 3]) b torch.Size([1, 3])
AffineNet flow torch.Size([1, 3, 128, 128, 128]) A torch.Size([1, 3, 3]) W torch.Size([1, 3, 3]) b torch.Size([1, 3])
reconstruction:  torch.Size([1, 1, 128, 128, 128]) torch.Size([1, 3, 128, 128, 128])
_interpolate im: torch.Size([1, 1, 128, 128, 128])
_interpolate base: torch.Size([2097152])




## Testing
Use this command to obtain the testing results.
```python
python eval.py  --name=SDHNet  --model=SDHNet_lpba --dataset=brain --dataset_test=lpba  --iters=6 --local_rank=0 --data_path=/xx/xx/  --base_path=/xx/xx/
```

## Datasets and Pre-trained Models (Based on Cascade VTN)
We follow Cascade VTN to prepare the training and testing datasets, please refer to [Cascade VTN](https://github.com/microsoft/Recursive-Cascaded-Networks) for details.

The related [pretrained models](https://drive.google.com/drive/folders/1BpxkIzL_SrPuKdqC_buiINawNZVMqoWc?usp=share_link) are available, please refer to the testing command for evaluating.

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{zhou2023self,
  title={Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration},
  author={Zhou, Shenglong and Hu, Bo and Xiong, Zhiwei and Wu, Feng},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```

## Contact
Due to our further exploration of the self-distillation, the current repo does not involve the related part temporarily. 

Please be free to contact us by e-mail (slzhou96@mail.ustc.edu.cn) or WeChat (ZslBlcony) if you have any questions.

## Acknowledgements
We follow the functional implementation in [Cascade VTN](https://github.com/microsoft/Recursive-Cascaded-Networks), and the overall code framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT). 

Thanks a lot for their great contribution!


