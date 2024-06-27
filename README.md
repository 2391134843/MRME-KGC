# MRME-KGC
Source code of the paper "Multi-view Riemannian Manifolds Fusion Enhancement for Knowledge Graph Completion".This paper was submitted to TKDE'2024


## Requirements

```
pip install -r requirement.txt
```
Details such like:

* python>=3.8
* torch>=1.86 (for mixed precision training)
* tqdm
* geoopt
* sklearn

All experiments are run with 4 RTX3090(24GB) GPUs.

## Data Preparation
1. Due to the size limit of github, please download the dataset (FB15K-237, CN-100K, Kinships, WN18RR, YAGO3-10, UML.) from [google drive](https://drive.google.com/drive/folders/1JR9KMjALZ_lJvp1oMQoi6XF4RYhRbCbF?usp=sharing) and save it in the following format:
- MRME-KGC
    - model
        - manifolds
            - file1
            - file2
        - file3
        .....
    - src_data
        - CN-100K
        - FB237
        - Kinships
        - UML
        - WN18RR
	    - YAGO3-10

2.After completing the above steps, please run the "process_datasets.py" file in the model folder to generate KG data to be input into the model:
```
cd model/
python process_datasets.py
```
## How to Run
### FB15K-237 dataset
```
bash Run-FB237-100d.sh
```
### CN100K dataset
```
bash Run-CN-100K-100d.sh
```

### UML dataset
```
bash Run-UML-100d.sh
```
....
## Troubleshooting
1. I encountered "CUDA out of memory" when running the code.
We run experiments with 4 RTX2090(24GB) GPUs, please reduce the batch size if you don't have enough resources.

2. ModuleNotFoundError: No module named 'sklearn'
Please note that the command used to install the 'sklearn' package is: "pip install -U scikit-learn" instead of "pip install -U sklearn". Please pay attention to this problem!

## Drawing
If you want to draw pictures similar to the model pictures in the paper about hyperbolas and spheres, please refer to the "Draw a picture of hyperbolic space.ipynb" file.
```
vim Draw a picture of hyperbolic space.ipynb
```

## 🤝 Connection
If you have any questions, please contact my email <xltx_youxiang@qq.com> !