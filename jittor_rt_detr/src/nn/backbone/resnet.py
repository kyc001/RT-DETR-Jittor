"""ResNet Backbone for RT-DETR
Aligned with PyTorch version implementation
"""

import jittor as jt
import jittor.nn as nn
from collections import OrderedDict

def ensure_float32(x):
    """确保张量为float32类型"""
    if isinstance(x, jt.Var):
        return x.float32()
    else:
        return x

def get_activation(act, inplace=True):
    """获取激活函数"""
    if act is None:
        return nn.Identity()

    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'swish':
        return nn.Swish()
    elif act == 'silu':
        return nn.SiLU()
    else:
        return nn.ReLU()

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.act = act
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act_layer = get_activation(act) if act else None

    def execute(self, x):
        x = ensure_float32(x)
        x = ensure_float32(self.conv(x))
        x = ensure_float32(self.norm(x))
        if self.act_layer:
            x = ensure_float32(self.act_layer(x))
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 1, 1, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, stride, act=act)
        self.branch2c = ConvNormLayer(ch_out, ch_out * self.expansion, 1, 1)
        self.act = get_activation(act)

    def execute(self, x):
        x = ensure_float32(x)
        out = ensure_float32(self.branch2a(x))
        out = ensure_float32(self.branch2b(out))
        out = ensure_float32(self.branch2c(out))

        if self.shortcut:
            short = x
        else:
            short = ensure_float32(self.short(x))

        out = ensure_float32(out + short)
        out = ensure_float32(self.act(out))
        return out


class PResNet(nn.Module):
    def __init__(self, depth=50, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=False, pretrained=False):
        super().__init__()

        block_nums = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3]
        }[depth]

        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(c_in, c_out, k, s, act=act))
            for c_in, c_out, k, s, name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = Bottleneck

        # 计算输出通道数
        _out_channels = [block.expansion * v for v in ch_out_list]  # [256, 512, 1024, 2048]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage = []
            stage_num = i + 2  # stage从2开始
            for j in range(block_nums[i]):
                stride = 2 if j == 0 and stage_num != 2 else 1  # 第一个stage不降采样
                shortcut = False if j == 0 else True

                stage.append(block(ch_in, ch_out_list[i], stride, shortcut, act, variant))

                # 更新输入通道数
                if j == 0:
                    ch_in = ch_out_list[i] * block.expansion

            self.res_layers.append(nn.Sequential(*stage))

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

    def execute(self, x):
        x = ensure_float32(x)
        x = ensure_float32(self.conv1(x))
        x = ensure_float32(nn.max_pool2d(x, kernel_size=3, stride=2, padding=1))

        outs = []
        for i, layer in enumerate(self.res_layers):
            x = ensure_float32(layer(x))
            if i in self.return_idx:
                outs.append(x)

        return outs


def ResNet50(pretrained=False, **kwargs):
    """ResNet50 backbone"""
    return PResNet(depth=50, **kwargs)

# 为了兼容性，保留旧的接口
def ResNet(block, layers):
    """Legacy ResNet interface"""
    return PResNet(depth=50)

__all__ = ['PResNet', 'ResNet50', 'ResNet', 'Bottleneck', 'ConvNormLayer']
