# /usr/bin/python3
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="One Shot tuner get result script")
parser.add_argument(
    "--root", type=str, required=True, help="<root path e.g. /root/eval_tuner/save_path>"
)
opt = parser.parse_args()
for network in [
    "alexnet",
    "densenet-121",
    "efficientnet",
    "resnet-18",
    "mobilenetv2",
    "squeezenet_v1.0",
    "bert",
    "vgg-16",
]:
    for algorithm in ["ga", "sa"]:
        try:
            root_path = opt.root
            second = f"{root_path}/{network}/NCHW/1/{algorithm}/second.npy"
            endtoend = f"{root_path}/{network}/NCHW/1/{algorithm}/end2end.npy"
            second = np.mean(np.load(second))
            endtoend = np.load(endtoend)
            print(
                f"network {network} algorithm {algorithm} inference time {second}ms end2end {endtoend}"
            )
            print()
        except Exception as e:
            print(e)
