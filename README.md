# RAIN-GS: Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting
<a href="https://arxiv.org/abs/2403.09413"><img src="https://img.shields.io/badge/arXiv-2403.09413-%23B31B1B"></a>
<a href="https://ku-cvlab.github.io/RAIN-GS/ "><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<br>

This is our official implementation of the paper "Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting"!

by [Jaewoo Jung](https://crepejung00.github.io)<sup>:umbrella:</sup>, [Jisang Han](https://onground-korea.github.io/)<sup>:umbrella:</sup>, [Honggyu An](https://hg010303.github.io/)<sup>:umbrella:</sup>, [Jiwon Kang](https://github.com/loggerJK)<sup>:umbrella:</sup>, [Seonghoon Park](https://github.com/seong0905)<sup>:umbrella:</sup>, [Seungryong Kim](https://cvlab.korea.ac.kr)<sup>&dagger;</sup>

:umbrella:: Equal Contribution <br>
&dagger;: Corresponding Author
## Introduction
![](assets/teaser.png)<br>
We introduce a novel optimization strategy (**RAIN-GS**) for 3D Gaussian Splatting!

We show that our simple yet effective strategy consisting of **sparse-large-variance (SLV) random initialization**, **progressive Gaussian low-pass filter control**, and the **Adaptive Bound-Expanding Split (ABE-Split) algorithm** robustly guides 3D Gaussians to model the scene even when starting from random point cloud.

**❗️Update (2024/05/29):** We have updated our paper and codes which significantly improve our previous results! <br>
**😴 TL;DR** for our update is as follows:
- We added a modification to the original split algorithm of 3DGS which enables the Gaussians to model scenes further from the viewpoints! This new splitting algorithm is named Adaptive Bound-Expanding Split algorithm (**ABE-Split** algorithm).
- Now with our three key components (SLV initialization, progressive Gaussians low-pass filtering, ABE-Split), we perform **on-par or even better** compared to 3DGS trainied with SfM initialized point cloud.

- As RAIN-GS only requires the initial point cloud to be sparse (SLV initialization), we now additionally apply our strategy to **SfM/Noisy SfM point cloud** by choosing a sparse set of points from the point cloud.

For further details and visualization results, please check out our updated [paper](https://arxiv.org/abs/2403.09413) and our new [project page](https://ku-cvlab.github.io/RAIN-GS/).

## Installation
We implement **RAIN-GS** above the official implementation of 3D Gaussian Splatting. <br> For environmental setup, we kindly guide you to follow the original requirements of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 

## Training

To train 3D Gaussians Splatting with our **updated** **RAIN-GS** novel strategy, all you need to do is:

```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval --ours_new 
```
You can train from various initializations by adding `--train_from ['random', 'reprojection', 'cluster', 'noisy_sfm']` (random is default)
<details>
<summary>Toggle to find more details for training from various initializations.</summary>

- **Random Initialization** (Default)
```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval --ours_new --train_from 'random'
```
- SfM (Structure-from-Motion) Initialization <br>
In order to apply RAIN-GS to SfM Initialization, we need to start with a sparse set of points (SLV Initialization). <br>
To choose the sparse set of points, you can choose several options:
  - **Clustering** : Apply clustering to the initial point cloud using the [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) algorithm.
  ```bash
  python train.py -s {dataset_path} --exp_name {exp_name} --eval --ours_new --train_from 'cluster'
  ```

  - **Top 10%** : Each of the points from SfM comes with a confidence value, which is the reprojection error. Select the top 10% most confident points from the point cloud.
  ```bash
  python train.py -s {dataset_path} --exp_name {exp_name} --eval --ours_new --train_from 'reprojection'
  ```

- **Noisy SfM Initialization** <br>
In real-world scenarios, the point cloud from SfM can contain noise. To simulate this scenario, we add a random noise sampled from a normal distribution to the SfM point cloud. If you run with this option, we apply the clustering algorithm to the Noisy SfM point cloud.
```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval --ours_new --train_from 'noisy_sfm'
```

</details>

To train 3D Gaussian Splatting with our original **RAIN-GS**, all you need to do is:

```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval --ours
```

For dense-small-variance (DSV) random initialization (used in the original 3D Gaussian Splatting), you can simply run with the following command:
```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval --paper_random
```

For SfM (Structure-from-Motion) initialization (used in the original 3D Gaussian Splatting), you can simply run with the following command:
```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval
```

For Noisy SfM initialization (used in the original 3D Gaussian Splatting), you can simply run with the following command:
```bash
python train.py -s {dataset_path} --exp_name {exp_name} --eval --train_from 'noisy_sfm'
```

To train with Mip-NeRF360 dataset, you can add argument `--images images_4` for outdoor scenes and `--images images_2` for indoor scenes to modify the resolution of the input images.

## Acknowledgement

We would like to acknowledge the contributions of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for open-sourcing the official codes for 3DGS! 

## Citation
If you find our work helpful, please cite our work as:
```
@article{jung2024relaxing,
  title={Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting},
  author={Jung, Jaewoo and Han, Jisang and An, Honggyu and Kang, Jiwon and Park, Seonghoon and Kim, Seungryong},
  journal={arXiv preprint arXiv:2403.09413},
  year={2024}
}
```
