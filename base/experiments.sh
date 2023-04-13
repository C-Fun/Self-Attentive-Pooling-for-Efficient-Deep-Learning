# STL10
python ann.py --dataset STL10 --batch_size 8 --architecture mobilenet_v2-nlp_headfix2-1222121 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
python ann.py --dataset STL10 --batch_size 8 --architecture mobilenet_v2-nlp_headfix2-2222121 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
python ann.py --dataset STL10 --batch_size 8 --architecture mobilenet_v2-nlp_headfix2-4222121 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log

python ann.py --dataset STL10 --batch_size 8 --architecture resnet18-nlp-1222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
python ann.py --dataset STL10 --batch_size 8 --architecture resnet18-nlp_headfix2-2222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
python ann.py --dataset STL10 --batch_size 8 --architecture resnet18-nlp_headfix2-4222 --im_size 128 --learning_rate 1e-2 --epochs 300 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1 --seed 0 --log
