# MutualForce
This is a repo of [MutualForce] for 3D object detection.

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and InterFusion.


## Overview
* [Introduction](#introduction)
* [Changelog](#changelog)
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

## Changelog



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

```
