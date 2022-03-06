#! /bin/bash
# with simulated annealing batch size 1
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network resnet-18
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network vgg-16
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network squeezenet_v1.0
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network densenet-121
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network mobilenetv2
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network efficientnet
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network alexnet
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer sa  --batch 1 --network bert

# with genetic algorithm batch size 1
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network resnet-18
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network vgg-16
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network squeezenet_v1.0
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network densenet-121
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network mobilenetv2
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network efficientnet
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network alexnet
python3 eval_tuner/tune.py --model_path `pwd`/train_model/model_best.pth --optimizer ga  --batch 1 --network bert