# Non-Local-Pooling



## STL10

- need to specify variable 'root' in ann.py

```shell
[base/ann.py]
...
root = '/nas/home/fangc/' # specify your home root
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

- for resnet you can also use dy+backbone to use dynamic convolution (e.g., dyresnet18, dymobilenet)

###### optional pooling methods:

​	*(skip, maxp, avgp, lip, gaussian_pool, nlp, dfmnlp, mixp, dfmixp)*

- for (nlp, dfmnlp, mixp, dfmixp), you can use pooling_headfix2 or pooling_reduced (e.g., nlp_headfix2, nlp_reduced)

Architecture example: prepool-nlp_headfix2-4-resnet18-nlp_headfix2-1222

###### Examples:

```shell
[Non-Local-Pooling/]
> cd base
[Non-Local-Pooling/base/]
> python ann.py --dataset STL10 --batch_size 8 --architecture RESNET18-SKIP-1222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture PREPOOL-NLP_HEADFIX2-4-RESNET18-SKIP-1222 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture MOBILENET-SKIP-2121 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture MOBILENET_v2-GAUSSIAN_POOL-2121 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
```

## MMDetection

- Firstly, you need to set your mmdetection library path:

```shell
[mmdet_train.sh]
#!/usr/bin/env bash
MMDET_ROOT='/nas/home/fangc/mmdetection/' # set your mmdetection library path here
...
```

```shell
[mmdet_dist_train.sh]
#!/usr/bin/env bash
MMDET_ROOT='/nas/home/fangc/mmdetection/' # set your mmdetection library path here
...
```

- Specify your coco data root:

```shell
[mmdet_repo/configs/common/coco_detection.py]
root = '/nas/home/fangc/' # specify your home root
...
data_root = root + '/data/coco/'
```

Then, run the shell file as follows:

```shell
[Non-Local-Pooling/]
# single-gpu
> sh mmdet_train.sh ./mmdet_repo/configs/network/faster_rcnn/dyresnet50.py
# multi-gpu
> sh mmdet_dist_train.sh ./mmdet_repo/configs/network/faster_rcnn/dyresnet50.py 4
```

