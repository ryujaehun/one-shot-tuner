to cite One-shot tuner:
``` bibtex
@inproceedings{10.1145/3497776.3517774,
author = {Ryu, Jaehun and Park, Eunhyeok and Sung, Hyojin},
title = {One-Shot Tuner for Deep Learning Compilers},
year = {2022},
isbn = {9781450391832},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3497776.3517774},
doi = {10.1145/3497776.3517774},
abstract = {Auto-tuning DL compilers are gaining ground as an optimizing back-end for DL frameworks. While existing work can generate deep learning models that exceed the performance of hand-tuned libraries, they still suffer from prohibitively long auto-tuning time due to repeated hardware measurements in large search spaces. In this paper, we take a neural-predictor inspired approach to reduce the auto-tuning overhead and show that a performance predictor model trained prior to compilation can produce optimized tensor operation codes without repeated search and hardware measurements. To generate a sample-efficient training dataset, we extend input representation to include task-specific information and to guide data sampling methods to focus on learning high-performing codes. We evaluated the resulting predictor model, One-Shot Tuner, against AutoTVM and other prior work, and the results show that One-Shot Tuner speeds up compilation by 2.81x to 67.7x compared to prior work while providing comparable or improved inference time for CNN and Transformer models.},
booktitle = {Proceedings of the 31st ACM SIGPLAN International Conference on Compiler Construction},
pages = {89–103},
numpages = {15},
keywords = {deep neural networks, autotuning, performance models, optimizing compilers},
location = {Seoul, South Korea},
series = {CC 2022}
}
```


# Install TVM
```
git clone --recursive https://github.com/ryujaehun/one-shot-tuner.git ost
cd one-shot-tuner
```

## To install the these minimal pre-requisites

```
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools python3-pip gcc libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev libjpeg-dev llvm llvm-10 llvm-10-dev clang-10 git
pip3 install cmake 
```

Edit build/config.cmake to customize the compilation options
```
mkdir build
cp cmake/config.cmake build
```
Change set(USE_CUDA OFF) to set(USE_CUDA ON) to enable CUDA backend
(e.g. https://gist.github.com/ryujaehun/5c841d3f5a7f720a14a3a7eb05326176)

## build tvm

```
cd build
cmake ..
make -j $(($(nproc) + 1))
cd ..
```

## set the environment variable
Append `~/.bashrc.`
```
export TVM_HOME=/path/to/one-shot-tuner
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```


## Python dependencies
```
pip3 install tornado psutil xgboost cloudpickle decorator pytest
pip3 install  -r requirements.txt
```

#  prior-guided task sampilng(PBS) and Exploration Based code Sampling(EBS)
The extracted dataset for CUDA data is included.

- `-p` activates Prior Guided Task Sampling.
- `-e` activates Exploration Based code sampling.

```
python3 dataset_generate/sampling.py -p -e
```

# Training a cost model 

```
python3 train_model/train.py --dataset_dir <dataset path (e.g. /root/ost/dataset_generate)> --layout NCHW --batch 1
```

#  Evaluating and collecting results

You can run all main experiment(CUDA device,NCHW format batch 1) using `main.sh` script.

```
main.sh
```

Create a folder using the save path and parameters that can be specified in the script and save the result (second,flops/s and end-to-end time).

__Example path__

- `<one-shot-tuner>/eval_tuner/save_path/resnet-18/NCHW/1/sa/flops.npy`

__How to collect results__

```
python3 get_result.py  
```

## Docker guide

If setting the environment is difficult, try using Docker container

```

docker run -it --rm --gpus 1 --name test jaehun/ost:v2 bash # docker running
cd /root/tvm
./main.sh 									  # start experiment
python3 get_result.py                         # get results
```

## Hardware dependencies

We recommend systems with NVIDIA GeForce RTX 2080
Ti GPU and Intel Xeon CPU E5-2666 v3 CPU (AWS c4 4x
large instances) for verifying GPU and CPU results respec-
tively. 

## Software dependencies

Our code is implemented and tested on Ubuntu 18.04 x86-64
system, with CUDA 10.2 and cudnn 7. Additional software
dependencies include minimal pre-requisites on Ubuntu for
TVM and deep learning frameworks, i.e., PyTorch v1.6.0,
for model implementations. We highly recommend using
the following docker image, ”jaehun/ost:v2”. 
