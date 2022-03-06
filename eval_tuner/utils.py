from tvm.relay import testing
import tvm
from tvm import relay
import os
import transformers
from transformers import *
import torch
import pickle
from transformers import logging

logging.set_verbosity_warning()


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    from mxnet.gluon.model_zoo.vision import get_model
    import tvm
    from tvm import relay
    from gluoncv2.model_provider import get_model as glcv2_get_model

    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if "resnet" in name:

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif "densenet" in name:
        mod, params = relay.testing.densenet.get_workload(batch_size=batch_size, dtype=dtype)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif "alexnet" in name:
        block = get_model("alexnet", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            dtype=dtype,
        )
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif name == "mobilenetv2":
        block = get_model("mobilenetv2_1.0", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, dtype=dtype, layout=layout
        )
    elif name == "bert":
        from transformers import BertTokenizer
        from transformers import BertConfig
        from transformers import BertModel

        enc = BertTokenizer.from_pretrained("bert-base-uncased")
        # Tokenizing input text
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = enc.tokenize(text)

        # Masking one of the input tokens
        masked_index = 8
        tokenized_text[masked_index] = "[MASK]"
        indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Creating a dummy input
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        dummy_input = [tokens_tensor, segments_tensors]

        # Initializing the model with the torchscript flag
        # Flag set to True even though it is not necessary as this model does not have an LM Head.
        config = BertConfig(
            vocab_size_or_config_json_file=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            torchscript=True,
        )
        # Instantiating the model
        model = BertModel(config)
        # The model needs to be in evaluation mode
        model.eval()
        # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
        model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
        # Creating the trace
        traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])

        shape_list = [
            (i.debugName().split(".")[0], i.type().sizes())
            for i in list(traced_model.graph.inputs())[1:]
        ]

        mod, params = tvm.relay.frontend.pytorch.from_pytorch(
            traced_model, shape_list, default_dtype="float32"
        )
        input_shape = tokens_tensor.numpy()
        output_shape = segments_tensors.numpy()
    elif name == "squeezenet_v1.0":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size,
            version="1.0",
            dtype=dtype,
        )
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "efficientnet":

        block = net = glcv2_get_model("EfficientNet_B0", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    elif name == "mxnet":
        # an example for mxnet model
        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape
