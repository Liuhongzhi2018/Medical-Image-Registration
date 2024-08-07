# Perceptual IQA Dataset (PIPAL) and Codebase

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

```bash 

conda env create -f environment.yaml -n myenv

```


## How to Train
- **Train model**
	1. Run command:

	```bash

	CUDA_VISIBLE_DEVICES=1  python codes/train_PIPAL.py -opt codes/options/train_test_yml/train_IQA_PIPAL_part.yml
	
	CUDA_VISIBLE_DEVICES=1  python codes/train_fIQA.py -opt codes/options/train_test_yml/train_IQA_fIQA_part.yml

	```

## How to Test
- **Test model**
	1. Run command:

	```bash

	cd codes/

	python test_PIPAL.py -opt options/train_test_yml/test_IQA_PIPAL_part_noGPU.yml

	python test_fIQA.py -opt options/train_test_yml/test_IQA_fIQA_noGPU.yml

	```

## Ackowledgement
- This code is based on [PIPAL](https://github.com/HaomingCai/PIPAL-dataset).