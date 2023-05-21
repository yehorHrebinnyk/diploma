import torch
import torch.nn as nn
import math

from pathlib import Path
from copy import deepcopy

from model.components.layers import Conv, Concat, SPPCSPC, MP, IDetect
from model.utils.general import make_divisible


def build_model(structure, channels):
    anchors, \
    number_classes, \
    depth_multiple, \
    width_multiple = structure["anchors"], \
                     structure["number_classes"], \
                     structure["depth_multiple"], \
                     structure["width_multiple"]

    number_anchors = (len(anchors[0]) // 2)
    number_outputs = number_anchors * (number_classes + 5)
    layers, save_outputs, channel_out = [], [], channels[-1]

    for i, (from_, number, module, args) in enumerate(structure["backbone"] + structure["head"]):
        module = eval(module) if isinstance(module, str) else module
        for j, arg in enumerate(args):
            try:
                args[j] = eval(arg) if isinstance(arg, str) else arg
            except:
                pass

        number = max(round(number * depth_multiple), 1) if number > 1 else number
        if module in [Conv, SPPCSPC]:
            channel_in, channel_out = channels[from_], args[0]
            if channel_out != number_outputs:
                channel_out = make_divisible(channel_out * width_multiple, 8)

            args = [channel_in, channel_out, *args[1:]]
            if module in [SPPCSPC]:
                args.insert(2, number)
                number = 1
        elif module is Concat:
            channel_out = sum([channels[x] for x in from_])
        elif module is IDetect:
            args.append([channels[x] for x in from_])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(from_)
        else:
            channel_out = channels[from_]

        sub_modules = nn.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)
        number_parameters = sum([x.numel() for x in sub_modules.parameters()])
        sub_modules.i, sub_modules.from_, sub_modules.np = i, from_, number_parameters
        save_outputs.extend(
            x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        layers.append(sub_modules)
        if i == 0:
            channels = []

        channels.append(channel_out)
    return nn.Sequential(*layers), save_outputs


class Yolo(nn.Module):
    def __init__(self, cfg='./model/config/yolov7x.yaml', channels=3, number_classes=None, anchors=None):
        super(Yolo, self).__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        channels = self.yaml.get("ch", channels)
        if number_classes and number_classes != self.yaml['number_classes']:
            self.yaml['number_classes'] = number_classes

        if anchors:
            self.yaml['anchors'] = round(anchors)

        self.model, self.save_outputs = build_model(deepcopy(self.yaml), channels=[channels])

        last_layer = self.model[-1]

        if isinstance(last_layer, IDetect):
            stride = 256
            last_layer.stride = torch.tensor(
                [stride / x.shape[-2] for x in self.forward(torch.zeros(1, channels, stride, stride))[:4]])
            last_layer.anchors /= last_layer.stride.view(-1, 1, 1)
            self.stride = last_layer.stride
            self._initialize_biases()

    def forward(self, x):
        module_outputs = []

        for module in self.model:
            if module.from_ != -1:
                x = module_outputs[module.from_] if isinstance(module.from_, int) \
                    else [x if j == -1 else module_outputs[j] for j in module.from_]

            x = module(x)
            module_outputs.append(x if module.i in self.save_outputs else None)

        return x

    def _initialize_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        last_layer = self.model[-1]  # Detect() module
        for mi, s in zip(last_layer.conv_outputs, last_layer.stride):  # from
            b = mi.bias.view(last_layer.number_anchors, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (last_layer.number_classes - 0.99)) if cf is None else torch.log(
                cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


if __name__ == "__main__":
    model = Yolo(number_classes=13)
    fake_img = torch.zeros((1, 3, 640, 640))
    res = model(fake_img)
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)
