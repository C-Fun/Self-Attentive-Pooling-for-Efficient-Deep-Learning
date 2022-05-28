# Non-Local-Pooling



## STL10

```shell
[Non-Local-Pooling/]
> cd base
[Non-Local-Pooling/base/]
> python ann.py --dataset STL10 --batch_size 8 --architecture RESNET50_NLP --im_size 224 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
```

## MMDetection

```shell
[Non-Local-Pooling]
> sh mmdet_dist_train.sh ./mmdetection/REPO/configs/network/faster_rcnn/dyresnet50.py 4
```

