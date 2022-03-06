import numpy as np
from torch.utils.data import Dataset
import os
import pickle
import glob
import pandas as pd
import torch
import ast


def config(kind, N, size, CO, CI, kernels, strides, padding, layout="NHWC"):
    import tvm
    from tvm.autotvm import env
    from tvm.autotvm import feature
    from tvm import relay
    from tvm import autotvm

    dtype = "float32"
    if kind == "depthwise":
        kind = "depthwise_conv2d_nchw.cuda"
    elif kind == "conv2d":
        kind = "conv2d_nchw.cuda"
    elif kind == "winograd":
        kind = "conv2d_nchw_winograd.cuda"
    name = kind
    if kind in [
        "conv2d_nchw_winograd.cuda",
        "conv2d_nchw.cuda",
        "conv2d_nhwc_tensorcore.cuda",
        "conv2d_nhwc.cuda",
        "conv2d_nhwc_winograd_direct.cuda",
        "conv2d_nhwc_winograd_tensorcore.cuda",
    ]:
        kind = "conv2d"
    elif kind in ["depthwise_conv2d_nchw.cuda"]:
        kind = "depthwise"

    if layout == "NCHW":
        data_layout = "NCHW"
        kernel_layout = "OIHW"
        if kind == "conv2d" or kind == "winograd":
            data = relay.var("data", shape=(N, CI, *size))
            kernel = relay.var("kernel", shape=(CO, CI, *kernels))
            kernel_shape = (CO, CI, *kernels)
        elif kind == "depthwise":
            data = relay.var("data", shape=(N, CI, *size))
            kernel = relay.var("kernel", shape=(CO, 1, *kernels))
            kernel_shape = (CO, 1, *kernels)
    elif layout == "NHWC":
        data_layout = "NHWC"
        if kind == "conv2d" or kind == "winograd":
            kernel_layout = "HWIO"
            data = relay.var("data", shape=(N, *size, CI))
            kernel = relay.var("kernel", shape=(*kernels, CI, CO))
            kernel_shape = (*kernels, CI, CO)
        elif kind == "depthwise":
            kernel_layout = "HWOI"
            data = relay.var("data", shape=(N, *size, CI))
            kernel = relay.var("kernel", shape=(*kernels, CO, 1))
            kernel_shape = (*kernels, CO, 1)
    if kind == "conv2d" or kind == "winograd":
        dilation = (1, 1)
        out = relay.nn.conv2d(
            data,
            kernel,
            strides=strides,
            padding=padding,
            dilation=dilation,
            channels=CO,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            kernel_size=(*kernels,),
            out_dtype=dtype,
        )
    elif kind == "depthwise":
        assert CO == CI  # depth-wise
        dilation = (1, 1)
        out = relay.nn.conv2d(
            data,
            kernel,
            groups=CO,
            strides=strides,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            padding=padding,
            dilation=dilation,
            channels=CO,
            kernel_size=(*kernels,),
            out_dtype=dtype,
        )
    ctx = tvm.gpu()
    mod = tvm.IRModule.from_expr(out)
    kernel_weights = tvm.nd.array(np.ones(kernel_shape, dtype=dtype), ctx)
    dict_params = {"kernel": kernel_weights}
    task = autotvm.task.extract_from_program(
        mod["main"],
        target="cuda",
        params=dict_params,
    )
    for _task in task:
        if _task.name == name:
            return _task


def feature_extract(index, task):
    import tvm
    from tvm.autotvm import env
    from tvm.autotvm import feature
    from tvm import relay
    from tvm import autotvm

    """extract feature for an index in extract_space"""

    env.GLOBAL_SCOPE.in_tuning = True

    config = task.config_space.get(index)
    with task.target:
        sch, args = task.instantiate(config)
    knobs = config.get_flatten_feature()[:20]
    knob_zeros = np.zeros(20)
    knob_zeros[-len(knobs) :] = knobs
    parameter = np.hstack(
        [list(map(lambda x: x.value, args[1].shape)), list(map(lambda x: x.value, args[2].shape))]
    )
    parameter_zeros = np.zeros(8)
    parameter_zeros[-len(parameter) :] = parameter
    curve = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=30, gpu_filter=False)
    curve_zeros = np.zeros(720)
    curve_zeros[-len(curve) :] = curve
    return np.hstack([curve_zeros, knob_zeros, parameter_zeros])


class PredictDataset(Dataset):
    MEAN = 0
    STD = 1

    def __init__(self, folder_path=None, layout="NCHW", cache=False):
        self.labels = []
        self.dataset = []
        self.cache = cache
        cnt = 0
        unpacked_file = os.path.join(f"{folder_path}", f"data.pkl")
        if os.path.exists(unpacked_file) and self.cache:
            with open(unpacked_file, "rb") as f:
                self.data_unpacked = pickle.load(f)
                for d, l in self.data_unpacked:
                    self.dataset.append(d)
                    self.labels.append(l)
        else:
            print(f"folder_path,{folder_path}")
            for item in glob.glob(f"{folder_path}/[0-9]*"):
                try:
                    if os.path.exists(f"{item}/random.log"):
                        df = pd.read_json(f"{item}/random.log", lines=True)
                        for inp, conf, out in zip(df["input"], df["config"], df["result"]):
                            N, _, size_h, size_w = inp[2][0][1]
                            CO, CI, kernel_h, kernel_w = inp[2][1][1]
                            size = (size_h, size_w)
                            kernels = (kernel_h, kernel_w)
                            strides = inp[2][4]
                            padding = inp[2][3]
                            kind = inp[1]
                            idx = conf["index"]
                            time = np.mean(out[0])
                            print(kind, N, size, CO, CI, kernels, strides, padding, layout)
                            task = config(kind, N, size, CO, CI, kernels, strides, padding, layout)
                            fea = feature_extract(idx, task)
                            flops = task.flop / time
                            self.dataset.append(fea)
                            self.labels.append(flops)
                        else:
                            print("not match")
                except Exception as e:
                    import traceback
                    import sys

                    traceback.print_exc()
                    print(
                        "Error on line {}".format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e
                    )
                    pass
                result = []
            for d, l in zip(self.dataset, self.labels):
                result.append([d, l])
            with open(unpacked_file, "wb") as f:
                pickle.dump(result, f)
        self.data_len = len(self.labels)
        self.mean = self.mean_get()
        self.std = self.std_get()
        self.set_value(self.mean, self.std)
        self.labels = np.array(self.labels)
        self.labels = self.normalize(self.labels)
        print(f"Dateset length {self.data_len}\tmean {self.mean}\tstd {self.std}")

    def __len__(self):
        return self.data_len

    def mean_get(self):
        return np.mean(self.labels)

    def std_get(self):
        return np.std(self.labels)

    @classmethod
    def set_value(cls, mean, std):
        cls.MEAN = mean
        cls.STD = std

    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx]), torch.tensor(self.labels[idx])


if "__main__" == __name__:
    import faulthandler

    faulthandler.enable()
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--layout", type=str, default="NHWC")
    parser.add_argument("--cache", action="store_true")
    opt = parser.parse_args()
    folder_path = f"{opt.layout}/{opt.batch}"
    PredictDataset(folder_path=folder_path, cache=opt.cache)
