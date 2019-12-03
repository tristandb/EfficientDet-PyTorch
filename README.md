# PyTorch EfficientDet
Here we implement [EfficientDet](https://arxiv.org/abs/1911.09070). The code is based on a RetinaNet implementation by [yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet). We use the EfficientNet backend by [rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch).

## Current status
Current implementation is able to run. I'll update this document as soon as I have some preliminary results. The paper by Tan et al. gives a few more details, which we would like to implement and report on:
* Add depthwise separable convolution for feature fusion.
* Use exponential moving average with decay 0.9998.
* Initialize convolution layers
* Train model using using SGD optimizer with momentum 0.9 and weight decay 4e-5.
* Implement described learning rate, which is first linearly increased from 0 to 0.08 in the initial 5% warm-up training steps and then annealed down using cosine decay rule. 
* Report performance.

If you have other issues that need my attention, feel free to make a pull request or leave an [issue](https://github.com/tristandb/EfficientDet-PyTorch/issues). 

## Results

Model | mAP | #Params | #FLOPS


## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install cffi

pip install pandas

pip install pycocotools

pip install cython

pip install opencv-python

pip install requests

pip install geffnet

```

4) Build the NMS extension.

```
cd pytorch-retinanet/lib
bash build.sh
cd ../
```

Note that you may have to edit line 14 of `build.sh` if you want to change which version of python you are building the extension for.

## Training

The network can be trained using the `train.py` script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use

```
python3 train.py --dataset coco --coco_path ../../Datasets/COCO2017 --phi 0 --batch-size 8
```

For training using a custom dataset, with annotations in CSV format (see below), use

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```

Note that the --csv_val argument is optional, in which case no validation will be performed.

## Acknowledgements
- The code is based on a RetinaNet implementation by [yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet). 
    - Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
    - The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)
- We use the EfficientNet backend by [rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch).