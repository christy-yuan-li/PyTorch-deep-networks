# PyTorch-deep-networks

This repository contains PyTorch implementation of several deep models. Training for the model is done using [TorchNet](https://github.com/pytorch/tnt)

## DenseNet with Atrous Convolutional Neural Network Architecture
This implements DenseNet enriched with Atrous Convolutional Operations in the last two Dense Blocks. The implementation follows [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) and [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

<img src="https://github.com/s1155026040/PyTorch-deep-networks/blob/master/figures/atrous1.png" alt="alt text" width=500 height=500>
<img src="https://github.com/s1155026040/PyTorch-deep-networks/blob/master/figures/atrous2.png" alt="alt text" width=500 height=500>
 

## DenseNet with Feature Pyramid Network Architecture 
This implements DenseNet enriched with Feature Pyramid Network Architecture. The implementation follows [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) and [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

<img src="https://github.com/s1155026040/PyTorch-deep-networks/blob/master/figures/fpn1.png" alt="alt text" width=500 height=500> 

## Capsule Network  
The implementation is based on [Dynamic Routing between Capsules](https://arxiv.org/abs/1710.09829) and an improvement of a PyTorch implementation [capsule-networks](https://github.com/gram-ai/capsule-networks.git)

<img src="https://github.com/s1155026040/PyTorch-deep-networks/blob/master/figures/capsule1.png" alt="alt text" width=500 height=500> 
<img src="https://github.com/s1155026040/PyTorch-deep-networks/blob/master/figures/capsule2.png" alt="alt text" width=500 height=500>  

The main source code is in "src" folder. The model files are in "models" folder. Change "src/train_capsule.py" accordingly to use different models. 
