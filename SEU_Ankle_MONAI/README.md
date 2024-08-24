<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

**M**edical **O**pen **N**etwork for **AI**

![Supported Python versions](https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/python.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/monai.svg)](https://badge.fury.io/py/monai)
[![docker](https://img.shields.io/badge/docker-pull-green.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/projectmonai/monai)
[![conda](https://img.shields.io/conda/vn/conda-forge/monai?color=green)](https://anaconda.org/conda-forge/monai)

[![premerge](https://github.com/Project-MONAI/MONAI/actions/workflows/pythonapp.yml/badge.svg?branch=dev)](https://github.com/Project-MONAI/MONAI/actions/workflows/pythonapp.yml)
[![postmerge](https://img.shields.io/github/checks-status/project-monai/monai/dev?label=postmerge)](https://github.com/Project-MONAI/MONAI/actions?query=branch%3Adev)
[![Documentation Status](https://readthedocs.org/projects/monai/badge/?version=latest)](https://docs.monai.io/en/latest/)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAI/branch/dev/graph/badge.svg?token=6FTC7U1JJ4)](https://codecov.io/gh/Project-MONAI/MONAI)

MONAI is a [PyTorch](https://pytorch.org/)-based, [open-source](https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE) framework for deep learning in healthcare imaging, part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/).
Its ambitions are:
- developing a community of academic, industrial and clinical researchers collaborating on a common foundation;
- creating state-of-the-art, end-to-end training workflows for healthcare imaging;
- providing researchers with the optimized and standardized way to create and evaluate deep learning models.


## Features
> _Please see [the technical highlights](https://docs.monai.io/en/latest/highlights.html) and [What's New](https://docs.monai.io/en/latest/whatsnew.html) of the milestone releases._

- flexible pre-processing for multi-dimensional medical imaging data;
- compositional & portable APIs for ease of integration in existing workflows;
- domain-specific implementations for networks, losses, evaluation metrics and more;
- customizable design for varying user expertise;
- multi-GPU multi-node data parallelism support.


## Installation

To install [the current release](https://pypi.org/project/monai/), you can simply run:

```bash
pip install monai
```

Please refer to [the installation guide](https://docs.monai.io/en/latest/installation.html) for other installation options.

```bash
# To install an editable version of MONAI, it is recommended to clone the codebase directly:
git clone https://github.com/Project-MONAI/MONAI.git

# This command will create a MONAI/ folder in your current directory. You can install it by running:
cd MONAI/
python setup.py develop

# or, to build with MONAI C++/CUDA extensions and install:
cd MONAI/
BUILD_MONAI=1 python setup.py develop
# for MacOS
BUILD_MONAI=1 CC=clang CXX=clang++ python setup.py develop

# The C++/CUDA extension features are currently experimental, a pre-compiled version is made available via the recent docker image releases. Building the extensions from source may require Ninja and CUDA Toolkit. By default, CUDA extension is built if torch.cuda.is_available(). It’s possible to force building by setting FORCE_CUDA=1 environment variable.

# You can verify the installation by:
python -c "import monai; monai.config.print_config()"

# Installing the recommended dependencies
pip install -e '.[nibabel,skimage]'

# Alternatively, to install all optional dependencies:
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -e ".[all]"

# To install all optional dependencies with pip based on MONAI development environment settings:
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -r requirements-dev.txt

```
<!-- (monai) liuhongzhi@user-SYS-7049GP-TRT:/mnt/lhz/Github/SEU_Ankle_MONAI$ python -c "import monai; monai.config.print_config()"

MONAI version: 0+unknown
Numpy version: 1.26.0
Pytorch version: 2.0.1
MONAI flags: HAS_EXT = True, USE_COMPILED = False, USE_META_DICT = False
MONAI rev id: None
MONAI __file__: /mnt/lhz/Github/SEU_Ankle_MONAI/monai/__init__.py

Optional dependencies:
Pytorch Ignite version: 0.4.11
ITK version: 5.3.0
Nibabel version: 5.1.0
scikit-image version: 0.22.0
scipy version: 1.11.3
Pillow version: 10.0.1
Tensorboard version: 2.15.1
gdown version: 4.7.1
TorchVision version: 0.15.2
tqdm version: 4.66.1
lmdb version: 1.4.1
psutil version: 5.9.6
pandas version: 2.1.3
einops version: 0.7.0
transformers version: 4.21.3
mlflow version: 2.8.1
pynrrd version: 1.0.0
clearml version: 1.13.2

For details about installing the optional dependencies, please visit:
    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies -->


<!-- (monai) liuhongzhi@user-SYS-7049GP-TRT:/mnt/lhz/Github/MONAI$ python -c "import monai; monai.config.print_config()"

MONAI version: 1.3.3rc1+9.gf1ef3e88
Numpy version: 1.26.0
Pytorch version: 2.0.1
MONAI flags: HAS_EXT = True, USE_COMPILED = False, USE_META_DICT = False
MONAI rev id: f1ef3e88f4bc594779c0c4188883f78bf6d1efff
MONAI __file__: /mnt/lhz/Github/MONAI/monai/__init__.py

Optional dependencies:
Pytorch Ignite version: 0.4.11
ITK version: 5.3.0
Nibabel version: 5.1.0
scikit-image version: 0.22.0
scipy version: 1.11.3
Pillow version: 10.0.1Tensorboard version: 2.15.1
gdown version: 4.7.1
TorchVision version: 0.15.2
tqdm version: 4.66.1
lmdb version: 1.4.1
psutil version: 5.9.6
pandas version: 2.1.3
einops version: 0.7.0
transformers version: 4.21.3
mlflow version: 2.8.1
pynrrd version: 1.0.0clearml version: 1.13.2

For details about installing the optional dependencies, please visit:
    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies -->


## Getting Started

[MedNIST demo](https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe) and [MONAI for PyTorch Users](https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T) are available on Colab.

Examples and notebook tutorials are located at [Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials).

Technical documentation is available at [docs.monai.io](https://docs.monai.io).

## Citation

If you have used MONAI in your research, please cite us! The citation can be exported from: https://arxiv.org/abs/2211.02701.

## Model Zoo
[The MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo) is a place for researchers and data scientists to share the latest and great models from the community.
Utilizing [the MONAI Bundle format](https://docs.monai.io/en/latest/bundle_intro.html) makes it easy to [get started](https://github.com/Project-MONAI/tutorials/tree/main/model_zoo) building workflows with MONAI.

## Contributing
For guidance on making a contribution to MONAI, see the [contributing guidelines](https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md).

## Community
Join the conversation on Twitter/X [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Ask and answer questions over on [MONAI's GitHub Discussions tab](https://github.com/Project-MONAI/MONAI/discussions).

## Links
- Website: https://monai.io/
- API documentation (milestone): https://docs.monai.io/
- API documentation (latest dev): https://docs.monai.io/en/latest/
- Code: https://github.com/Project-MONAI/MONAI
- Project tracker: https://github.com/Project-MONAI/MONAI/projects
- Issue tracker: https://github.com/Project-MONAI/MONAI/issues
- Wiki: https://github.com/Project-MONAI/MONAI/wiki
- Test status: https://github.com/Project-MONAI/MONAI/actions
- PyPI package: https://pypi.org/project/monai/
- conda-forge: https://anaconda.org/conda-forge/monai
- Weekly previews: https://pypi.org/project/monai-weekly/
- Docker Hub: https://hub.docker.com/r/projectmonai/monai
