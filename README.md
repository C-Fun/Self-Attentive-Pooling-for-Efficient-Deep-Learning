# Self-Attentive Pooling for Efficient Deep Learning

Official PyTorch implementation of the paper entitled 'Self Attentive Pooling for Efficient Deep Learning'.

## Image Recognition

- need to specify variable 'root' in ann.py

```shell
[base/ann.py]
...
root = '/home/$username/' # specify your home root
```

- create a 'data' folder at the home root

```shell
home root
|- data
|- |- cifar100
|- |- stl10
|- |- coco
|- Non-Local-Pooling
```

#### Architecture is specified by keywords: 

​	***[prepool-pooling-stride]-backbone-pooling-strides***

###### optional backbones: 

​	*(resnet18, resnet18_v2, resnet50, resnet50_v2, mobilenet, mobilenet_v2)*

- you can also use dy+backbone to use dynamic convolution (e.g., dyresnet18, dymobilenet)

###### optional pooling methods:

​	*(skip, maxp, avgp, lip, gaussian_pool, nlp, dfmnlp, mixp, dfmixp)*

- for (nlp, dfmnlp, mixp, dfmixp), you can use pooling_headfix2, pooling_nowin or pooling_reduced (e.g., nlp_headfix2_nowin (recommended), nlp_reduced)

Architecture example: 

- prepool-nlp_headfix2-4-resnet18-nlp_headfix2-1222
- mobilenet_v2-gaussian_pool-4222121

###### Examples:

```shell
[Non-Local-Pooling/]
> cd base
[Non-Local-Pooling/base/]
> python ann.py --dataset STL10 --batch_size 8 --architecture mobilenet_v2-nlp_headfix2-1222121 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
> python ann.py --dataset STL10 --batch_size 8 --architecture mobilenet_v2-nlp_headfix2-2222121 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
> python ann.py --dataset STL10 --batch_size 8 --architecture mobilenet_v2-nlp_headfix2-4222121 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture resnet18-nlp-1222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
> python ann.py --dataset STL10 --batch_size 8 --architecture resnet18-nlp_headfix2-2222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
> python ann.py --dataset STL10 --batch_size 8 --architecture resnet18-nlp_headfix2-4222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
```

## Object Detection

- Firstly, you need to set your mmdetection library path:

```shell
[mmdet_train.sh]
#!/usr/bin/env bash
MMDET_ROOT='/home/$username/mmdetection/' # set your mmdetection library path here
...
```

```shell
[mmdet_dist_train.sh]
#!/usr/bin/env bash
MMDET_ROOT='/home/$username//mmdetection/' # set your mmdetection library path here
...
```

- Specify your coco data root:

```shell
[mmdet_repo/configs/common/coco_detection.py]
root = '/home/$username/' # specify your home root
...
data_root = root + '/data/coco/'
```

Then, run the shell file as follows:

```shell
[Non-Local-Pooling/]
# single-gpu
> sh mmdet_train.sh ./mmdet_repo/configs/network/faster_rcnn/resnet18/resnet18-nlp.py
> sh mmdet_train.sh ./mmdet_repo/configs/network/faster_rcnn/resnet18/resnet18-nlp_reduced.py
> sh mmdet_train.sh ./mmdet_repo/configs/network/ssd/resnet18/ssd_resnet18-nlp-1222.py
> sh mmdet_train.sh ./mmdet_repo/configs/network/ssd/mobilenet/nlp-1222121.py

# multi-gpu
> sh mmdet_dist_train.sh ./mmdet_repo/configs/network/faster_rcnn/resnet18.py 4
```

## Citation

If you find this work useful for your research, please cite our [paper](https://www.computer.org/csdl/proceedings-article/wacv/2023/934600d963/1L6LypkixNe):

```
@INPROCEEDINGS {10030316,
author = {F. Chen and G. Datta and S. Kundu and P. A. Beerel},
booktitle = {2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
title = {Self-Attentive Pooling for Efficient Deep Learning},
year = {2023},
volume = {},
issn = {},
pages = {3963-3972},
abstract = {Efficient custom pooling techniques that can aggressively trim the dimensions of a feature map for resource-constrained computer vision applications have recently gained significant traction. However, prior pooling works extract only the local context of the activation maps, limiting their effectiveness. In contrast, we propose a novel non-local self-attentive pooling method that can be used as a drop-in replacement to the standard pooling layers, such as max/average pooling or strided convolution. The proposed self-attention module uses patch embedding, multihead self-attention, and spatial-channel restoration, followed by sigmoid activation and exponential soft-max. This self-attention mechanism efficiently aggregates dependencies between non-local activation patches during downsampling. Extensive experiments on standard object classification and detection tasks with various convolutional neural network (CNN) architectures demonstrate the superiority of our proposed mechanism over the state-of-the-art (SOTA) pooling techniques. In particular, we surpass the test accuracy of existing pooling techniques on different variants of MobileNet-V2 on ImageNet by an average of ~1.2%. With the aggressive down-sampling of the activation maps in the initial layers (providing up to 22x reduction in memory consumption), our approach achieves 1.43% higher test accuracy compared to SOTA techniques with iso-memory footprints. This enables the deployment of our models in memory-constrained devices, such as micro-controllers (without losing significant accuracy), because the initial activation maps consume a significant amount of on-chip memory for high-resolution images required for complex vision tasks. Our pooling method also leverages channel pruning to further reduce memory footprints. Codes are available at https://github.com/CFun/Non-Local-Pooling.},
keywords = {deep learning;computer vision;limiting;memory management;feature extraction;system-on-chip;image restoration},
doi = {10.1109/WACV56688.2023.00396},
url = {https://doi.ieeecomputersociety.org/10.1109/WACV56688.2023.00396},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {jan}
}
```