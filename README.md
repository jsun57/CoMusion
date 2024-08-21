# CoMusion
### CoMusion: Towards Consistent Stochastic Human Motion Prediction via Motion Diffusion (ECCV'2024)
by [Jiarui Sun](https://ece.illinois.edu/about/directory/grad-students/jsun57), and [Girish Chowdhary](https://ece.illinois.edu/about/directory/faculty/girishc)


[[Project Page](https://jsun57.github.io/CoMusion/)] [[Paper](https://arxiv.org/pdf/2305.12554)]

This repository contains the official PyTorch implementation of the paper:

We develop a single-stage diffusion-based stochastic HMP framework to predict accurate, realistic, and consistent motions with respect to motion history.

## Installation


### 1. Environment

<details> 
<summary>Python/conda/mamba environment</summary>
<p>

```
Coming Soon!
```
</p>
</details> 


### 2. Datasets

#### [**> Human3.6M**](http://vision.imar.ro/human3.6m/description.php)

We follow https://github.com/wei-mao-2019/gsps for Human3.6M dataset preparation. 
All data needed can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sb1n9l0Na5EqtapDVShOJJ-v6o-GZrIJ?usp=sharing) and place all the dataset in ``data`` folder inside the root of this repo.


#### [**> AMASS**](https://amass.is.tue.mpg.de/)
We follow https://github.com/BarqueroGerman/BeLFusion for AMASS dataset preparation.
Due to the distribution policy of AMASS dataset, we are not allowed to distribute the data directly.
Please reach out if you have questions.

## Evaluation
Run the following scripts to evaluate CoMusion.

Human3.6M:
```
python train.py --cfg h36m --test
```

AMASS:
```
python train.py --cfg amass --test
```

## Training
Run the following scripts to train CoMusion.

Human3.6M:
```
python train.py --cfg h36m
```

AMASS:
```
python train.py --cfg amass
```


## Citation
If you find our work useful in your research, please consider citing our paper:
```
Coming Soon!
```
**Note:** 
We thank [German Barquero](https://barquerogerman.github.io/) for the [BeLFusion code](https://github.com/BarqueroGerman/BeLFusion) and his prompt QA and support. We also borrow parts from [GSPS](https://github.com/wei-mao-2019/gsps) by [Wei Mao](https://wei-mao-2019.github.io/home/).