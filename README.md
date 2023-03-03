# S-DETR (Ship DEtection TRansformer)

This repository is an official PyTorch implementation of the paper "S-DETR: A Transformer model for real-time detection of marine ships". 

## Installation

### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- Linux
- 8x NVIDIA V100 GPUs (32GB)
- CUDA 10.1
- Python == 3.8
- PyTorch == 1.8.1+cu101, TorchVision == 0.9.1+cu101
- GCC == 7.5.0
- cython, pycocotools, tqdm, scipy

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work.

### Code Installation

install PyTorch and TorchVision:

(preferably using our recommended setups; CUDA version should match your own local environment)
```bash
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.1 -c pytorch
```

After that, install other requirements:
```bash
conda install cython scipy tqdm
```

### Data Preparation

Please download [SMD dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) and organize them as following:

```
code_root/
└── data/
    └── SMD/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```


## Usage

### Training
To perform training on SMD&MID , modify the arguments based on the scripts below:
```shell
python -m torch.distributed.launch \
    --nproc_per_node=4 \        # number of GPUs to perform training
    --use_env main.py \
    --batch_size 4 \            # batch_size on individual GPU (this is *NOT* total batch_size)
    --SDETR \                    # to integrate with SDETR, remove this line to disable SDETR
    --dilation \                # to enable DC5, remove this line to disable DC5
    --multiscale \              # to enable multi-scale, remove this line to disable multiscale
    --epochs 50 \               # total number of epochs to train
    --lr_drop 40 \              # when to drop learning rate
    --output_dir output/xxxx    # where to store outputs, remove this line for not storing outputs
```
More arguments and their explanations are available at ```main.py```.

### Evaluation
To evaluate a model on SMD *val2017*, simply add ```--resume``` and ```--eval``` arguments to your training scripts:
```shell
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --batch_size 4 \
    --SDETR \
    --dilation \                
    --multiscale \ 
    --epochs 50 \
    --lr_drop 40 \ 
    --resume <path/to/checkpoint.pth> \   # trained model weights
    --eval \                              # this means that only evaluation will be performed
    --output_dir output/xxxx   
```


### Visualize Detection Results
We provide `demo.py`, which is a minimal implementation that allows users to visualize model's detection predictions. It performs detection on images inside the `./images` folder, and stores detection visualizations in that folder. Taking <b>S-DETR-R50 w/ SDETR (50 epochs)</b> for example, simply run:
```shell
python demo.py \                       # do NOT use distributed mode
    --SDETR \
    --epochs 50 \                      # you need to set this correct. See models/fast_detr.py L50-79 for details.
    --resume <path/to/checkpoint.pth>  # trained model weights
```