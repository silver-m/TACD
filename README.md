# TACD: A Novel 3-D Swin Transformer With Enhanced Feature Aggregation for Change Detection in Image Time Series

This repository contains the PyTorch implementation of **TACD**, a 3-D Swin Transformer-based framework for change detection in image time series. The code is built on top of the [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and [MMAction2](https://github.com/open-mmlab/mmaction2) codebases, and adapts 3-D shifted-window Transformer modeling to remote sensing image time-series change detection.

## Repository Structure

```text
TACD-main/
├── configs/
│   └── recognition/swin/                       # Swin Transformer configuration files
├── dcn/                                        # Deformable convolution implementation
├── mmaction/
│   └── models/backbones/swin_transformer.py    # Core TACD / 3-D Swin Transformer backbone
├── tools/
│   └── CD/
│       ├── train_cd_224_OSCD_aug_Prepro_0921.py
│       ├── train_cd_224_S7_aug_Prepro_1018.py
│       ├── custom_OSCD_aug_online_Prepro.py
│       ├── custom_Space7_aug_online_Prepro.py
│       └── toolsmu.py

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/TACD.git
cd TACD
```

### 2. Create a conda environment

```bash
conda create -n tacd python=3.8 -y
conda activate tacd
```

### 3. Install PyTorch

Install the PyTorch version that matches your CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

### 4. Install MMCV / MMAction2 dependencies

This project is based on the older MMAction2 / MMCV framework. A compatible setup is recommended, for example:

```bash
pip install mmcv-full
pip install -r requirements.txt
pip install pandas scikit-image torchnet timm einops tqdm matplotlib opencv-python
```

If `mmcv-full` installation fails, please install the version matching your PyTorch and CUDA environment from the official OpenMMLab wheel index.

### 5. Compile deformable convolution

The model uses deformable convolution modules under `dcn/`. Compile them before training:

```bash
cd dcn
bash make.sh
cd ..
```

On Windows, you may refer to `dcn/make.bat`.

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@ARTICLE{10752673,
  author={Mao, Yin and He, Qiuhua and Li, Jianlong and Yang, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={TACD: A Novel 3-D Swin Transformer With Enhanced Feature Aggregation for Change Detection in Image Time Series}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2024.3496994}}
```
Please also consider citing the original Video Swin Transformer work:

```bibtex
@INPROCEEDINGS{9878941,
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Video Swin Transformer}, 
  year={2022},
  volume={},
  number={},
  pages={3192-3201},
  doi={10.1109/CVPR52688.2022.00320}}
```

## Acknowledgements

This repository is developed based on Video Swin Transformer and MMAction2. We thank the authors and contributors of these open-source projects.
