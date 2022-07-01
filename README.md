# Non-Local-Pooling



## STL10

```shell
[Non-Local-Pooling/]
> cd base
[Non-Local-Pooling/base/]
> python ann.py --dataset STL10 --batch_size 8 --architecture RESNET18-SKIP-1222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture PREPOOL-RESNET18-NLP_HEADFIX2-4 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture MOBILENET-SKIP-2121 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

> python ann.py --dataset STL10 --batch_size 8 --architecture MOBILENET_v2-GAUSSIAN_POOL-2121 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
```

## MMDetection

Firstly, you need to set your mmdetection library path:

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



Then, run the shell file as follows:

```shell
[Non-Local-Pooling/]
# single-gpu
> sh mmdet_train.sh ./mmdet_repo/configs/network/faster_rcnn/dyresnet50.py
# multi-gpu
> sh mmdet_dist_train.sh ./mmdet_repo/configs/network/faster_rcnn/dyresnet50.py 4
```

