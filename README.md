# MRME-KGC
Source code of the paper "Multi-view Riemannian Manifolds Fusion Enhancement for Knowledge Graph Completion". 


## Requirements

```
pip install -r requirement.txt
```
Details such like:

* python>=3.8
* torch>=1.8
* tqdm
* geoopt
* sklearn

All experiments are run with 4 RTX3090(24GB) GPUs.

## Data Preparation
Due to the size limit of github, please download the dataset (FB15K-237, CN-100K, Kinships, WN18RR, YAGO3-10, UML.) from [google drive](https://drive.google.com/drive/folders/1JR9KMjALZ_lJvp1oMQoi6XF4RYhRbCbF?usp=sharing) .



## How to Run

### CN100K dataset
```
bash Run-CN-100K-100d.sh
```

### UML dataset
```
bash Run-UML-100d.sh
```
....


## Drawing
If you want to draw pictures similar to the model pictures in the paper about hyperbolas and spheres, please refer to the "Draw a picture of hyperbolic space.ipynb" file.
```
vim Draw a picture of hyperbolic space.ipynb
```

