python ann.py --dataset STL10 --batch_size 8 --architecture RESNET18-SKIP-1222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

python ann.py --dataset STL10 --batch_size 8 --architecture PREPOOL-NLP_HEADFIX2-4-RESNET18-SKIP-1222 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

python ann.py --dataset STL10 --batch_size 8 --architecture MOBILENET-SKIP-1222121 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

python ann.py --dataset STL10 --batch_size 8 --architecture MOBILENET_v2-GAUSSIAN_POOL-4222121 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

