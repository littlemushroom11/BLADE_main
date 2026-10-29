from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNNClient(nn.Module):
    def __init__(self, in_channels: int = 3, split: str = 'middle'):
        super().__init__()
        self.split = split
        self.b1 = ConvBlock(in_channels, 32)
        self.b2 = ConvBlock(32, 64)
        self.b3 = ConvBlock(64, 128)

    def forward(self, x):
        x = self.b1(x)
        if self.split == 'early':
            return x
        x = F.max_pool2d(x, 2)
        x = self.b2(x)
        if self.split == 'middle':
            return x
        x = F.max_pool2d(x, 2)
        x = self.b3(x)
        if self.split == 'late':
            return x
        raise ValueError(f'Unknown split: {self.split}')


class SmallCNNServer(nn.Module):
    def __init__(self, split: str = 'middle', num_classes: int = 10):
        super().__init__()
        self.split = split
        if split == 'early':
            c_in = 32
            self.head = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(c_in, 64),
                nn.MaxPool2d(2),
                ConvBlock(64, 128),
                nn.AdaptiveAvgPool2d(1),
            )
            c_final = 128
        elif split == 'middle':
            c_in = 64
            self.head = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(c_in, 128),
                nn.AdaptiveAvgPool2d(1),
            )
            c_final = 128
        elif split == 'late':
            c_in = 128
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1))
            c_final = 128
        else:
            raise ValueError(f'Unknown split: {split}')
        self.fc = nn.Linear(c_final, num_classes)

    def forward(self, z):
        h = self.head(z).flatten(1)
        return self.fc(h)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


def _make_resnet_layer(in_planes: int, planes: int, num_blocks: int, stride: int) -> Tuple[nn.Sequential, int]:
    layers = []
    strides = [stride] + [1] * (num_blocks - 1)
    cur = in_planes
    for s in strides:
        layers.append(BasicBlock(cur, planes, s))
        cur = planes * BasicBlock.expansion
    return nn.Sequential(*layers), cur


class ResNetClient(nn.Module):
    def __init__(self, in_channels: int = 3, split: str = 'middle', layers=(2, 2, 2, 2)):
        super().__init__()
        self.split = split
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1, c1 = _make_resnet_layer(64, 64, layers[0], stride=1)
        self.layer2, c2 = _make_resnet_layer(c1, 128, layers[1], stride=2)
        self.layer3, c3 = _make_resnet_layer(c2, 256, layers[2], stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        if self.split == 'early':
            return x
        x = self.layer2(x)
        if self.split == 'middle':
            return x
        x = self.layer3(x)
        if self.split == 'late':
            return x
        raise ValueError(f'Unknown split: {self.split}')


class ResNetServer(nn.Module):
    def __init__(self, split: str = 'middle', num_classes: int = 10, layer_counts=(2, 2, 2, 2)):
        super().__init__()
        self.split = split
        layers = []
        if split == 'early':
            layer2, c2 = _make_resnet_layer(64, 128, layer_counts[1], stride=2)
            layer3, c3 = _make_resnet_layer(c2, 256, layer_counts[2], stride=2)
            layer4, c4 = _make_resnet_layer(c3, 512, layer_counts[3], stride=2)
            layers.extend([layer2, layer3, layer4])
            c_final = c4
        elif split == 'middle':
            layer3, c3 = _make_resnet_layer(128, 256, layer_counts[2], stride=2)
            layer4, c4 = _make_resnet_layer(c3, 512, layer_counts[3], stride=2)
            layers.extend([layer3, layer4])
            c_final = c4
        elif split == 'late':
            layer4, c4 = _make_resnet_layer(256, 512, layer_counts[3], stride=2)
            layers.append(layer4)
            c_final = c4
        else:
            raise ValueError(f'Unknown split: {split}')
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_final, num_classes)

    def forward(self, z):
        h = self.layers(z)
        h = self.avgpool(h).flatten(1)
        return self.fc(h)


class ResNet18Client(ResNetClient):
    def __init__(self, in_channels: int = 3, split: str = 'middle'):
        super().__init__(in_channels, split, layers=(2, 2, 2, 2))


class ResNet18Server(ResNetServer):
    def __init__(self, split: str = 'middle', num_classes: int = 10):
        super().__init__(split, num_classes, layer_counts=(2, 2, 2, 2))


class ResNet34Client(ResNetClient):
    def __init__(self, in_channels: int = 3, split: str = 'middle'):
        super().__init__(in_channels, split, layers=(3, 4, 6, 3))


class ResNet34Server(ResNetServer):
    def __init__(self, split: str = 'middle', num_classes: int = 200):
        super().__init__(split, num_classes, layer_counts=(3, 4, 6, 3))


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


def _make_mobilenetv2_stage(inp: int, oup: int, n: int, stride: int, expand_ratio: int = 6) -> Tuple[nn.Sequential, int]:
    layers = []
    cur = inp
    for i in range(n):
        s = stride if i == 0 else 1
        layers.append(InvertedResidual(cur, oup, s, expand_ratio))
        cur = oup
    return nn.Sequential(*layers), cur


class MobileNetV2Client(nn.Module):
    def __init__(self, in_channels: int = 3, split: str = 'middle'):
        super().__init__()
        self.split = split
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage16, c16 = _make_mobilenetv2_stage(32, 16, n=1, stride=1, expand_ratio=1)
        self.stage24, c24 = _make_mobilenetv2_stage(c16, 24, n=2, stride=2, expand_ratio=6)
        self.stage32, c32 = _make_mobilenetv2_stage(c24, 32, n=3, stride=2, expand_ratio=6)
        self.stage64, _ = _make_mobilenetv2_stage(c32, 64, n=4, stride=2, expand_ratio=6)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage16(x)
        x = self.stage24(x)
        if self.split == 'early':
            return x
        x = self.stage32(x)
        if self.split == 'middle':
            return x
        x = self.stage64(x)
        if self.split == 'late':
            return x
        raise ValueError(f'Unknown split: {self.split}')


class MobileNetV2Server(nn.Module):
    def __init__(self, split: str = 'middle', num_classes: int = 10):
        super().__init__()
        self.split = split
        layers = []
        if split == 'early':
            stage32, c32 = _make_mobilenetv2_stage(24, 32, n=3, stride=2, expand_ratio=6)
            stage64, c64 = _make_mobilenetv2_stage(c32, 64, n=4, stride=2, expand_ratio=6)
            layers.extend([stage32, stage64])
            cur = c64
        elif split == 'middle':
            stage64, c64 = _make_mobilenetv2_stage(32, 64, n=4, stride=2, expand_ratio=6)
            layers.append(stage64)
            cur = c64
        elif split == 'late':
            cur = 64
        else:
            raise ValueError(f'Unknown split: {split}')
        stage96, c96 = _make_mobilenetv2_stage(cur, 96, n=3, stride=1, expand_ratio=6)
        stage160, c160 = _make_mobilenetv2_stage(c96, 160, n=3, stride=2, expand_ratio=6)
        stage320, c320 = _make_mobilenetv2_stage(c160, 320, n=1, stride=1, expand_ratio=6)
        layers.extend([stage96, stage160, stage320])
        self.layers = nn.Sequential(*layers)
        self.final_conv = nn.Sequential(
            nn.Conv2d(c320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, z):
        h = self.layers(z)
        h = self.final_conv(h)
        h = self.avgpool(h).flatten(1)
        return self.fc(h)


def build_split_model(cfg: dict) -> Tuple[nn.Module, nn.Module]:
    arch = cfg['model'].get('arch', 'smallcnn').lower()
    in_channels = int(cfg['model']['in_channels'])
    num_classes = int(cfg['model']['num_classes'])
    split = cfg['model'].get('split', 'middle')
    if arch == 'smallcnn':
        return SmallCNNClient(in_channels, split), SmallCNNServer(split, num_classes)
    if arch in {'resnet18', 'resnet18_cifar', 'cifar_resnet18'}:
        return ResNet18Client(in_channels, split), ResNet18Server(split, num_classes)
    if arch in {'resnet34', 'resnet34_cifar', 'cifar_resnet34'}:
        return ResNet34Client(in_channels, split), ResNet34Server(split, num_classes)
    if arch in {'mobilenetv2', 'mobilenet_v2', 'mobilev2'}:
        return MobileNetV2Client(in_channels, split), MobileNetV2Server(split, num_classes)
    raise ValueError(f'Unsupported arch={arch}. Supported: smallcnn, resnet18, resnet34, mobilenetv2.')
