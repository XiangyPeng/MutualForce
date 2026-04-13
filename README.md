# MutualForce
This is a repo of [MutualForce] for 3D object detection.

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and InterFusion.


## Overview
* [Introduction](#introduction)
* [Dataset Structure](#dataset-structure)
* [Experiment Scenario](#experiment-scenario)
* [Experiment Settings](#experiment-settings)
* [Experiment Results](#experiment-results)
* [Installation](#installation)
* [Training](#training)
* [Testing](#testing)

## Introduction
MutualForce: Mutual-Aware Enhancement for 4D Radar-LiDAR 3D Object Detection
ICASSD review
* Model Framework:
<p align="center">
  <img src="docs/force.png" width="95%">
</p>


## Installation


b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```
c. Generate dataloader
```
python -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset.yaml
```

## Training
```
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/astyx_models/pointpillar.yaml --tcp_port 25851 --extra_tag yourmodelname
```

## Testing
```
python test.py --cfg_file cfgs/astyx_models/pointpillar.yaml --batch_size 4 --ckpt ##astyx_models/pointpillar/debug/ckpt/checkpoint_epoch_80.pth
```

## Citation 
If you find this project useful in your research, please consider cite:


```
@INPROCEEDINGS{10887748,
  author={Peng, Xiangyuan and Sun, Huawei and Bierzynski, Kay and Fischbacher, Anton and Servadei, Lorenzo and Wille, Robert},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MutualForce: Mutual-Aware Enhancement for 4D Radar-LiDAR 3D Object Detection}, 
  year={2025},
  doi={10.1109/ICASSP49660.2025.10887748}}
```
