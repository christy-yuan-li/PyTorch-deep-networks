import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
from PIL import Image
sys.path.append('../')
import transforms


__all__ = ['DenseNet_atrous_aspp', 'densenet121_atrous_aspp', 'densenet169_atrous_aspp', 'densenet201_atrous_aspp', 'densenet161_atrous_aspp']


def densenet121_atrous_aspp(**kwargs):
    model = DenseNet_atrous_aspp(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169_atrous_aspp(**kwargs):
    model = DenseNet_atrous_aspp(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201_atrous_aspp(**kwargs):
    model = DenseNet_atrous_aspp(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet161_atrous_aspp(**kwargs):
    model = DenseNet_atrous_aspp(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _DenseLayer_hdc(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation):
        super(_DenseLayer_hdc, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1*dilation, dilation=dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer_hdc, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock_hdc(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation):
        super(_DenseBlock_hdc, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer_hdc(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, dilation)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    # def __init__(self, num_input_features, num_output_features):
    #     super(_Transition, self).__init__()
    #     self.add_module('norm', nn.BatchNorm2d(num_input_features))
    #     self.add_module('relu', nn.ReLU(inplace=True))
    #     self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
    #     self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
    def __init__(self, num_input_features, num_output_features, dropRate=0.0):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.max_pool2d(out, kernel_size=2, stride=2)

class _ASPP(nn.Module):
    '''
    ASPP with image pooling features
    '''
    def __init__(self, num_input_features, rate=6):
        '''
        :param num_input_features:
        :param size:  (int, int)
        '''
        super(_ASPP, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.branch1 = nn.Conv2d(num_input_features, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch2 = nn.Conv2d(num_input_features, 256, kernel_size=3, stride=1, padding=rate*1, bias=False, dilation=rate*1)
        self.branch3 = nn.Conv2d(num_input_features, 256, kernel_size=3, stride=1, padding=rate*2, bias=False, dilation=rate*2)
        self.branch4 = nn.Conv2d(num_input_features, 256, kernel_size=3, stride=1, padding=rate*3, bias=False, dilation=rate*3)

        self.conv1 = nn.Conv2d(num_input_features, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256*5, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256 * 5)

    def forward(self, x):
        out = self.relu(self.bn(x))
        aspplist = []
        # aspp features
        aspplist.append(self.branch1(out))
        aspplist.append(self.branch2(out))
        aspplist.append(self.branch3(out))
        aspplist.append(self.branch4(out))
        # image pooling features
        pooled = F.avg_pool2d(out, kernel_size=out.size(3), stride=1)   # (B, D, 1, 1)
        pooled = self.conv1(pooled)
        pooled = nn.Upsample(size=(out.size(2), out.size(3)), mode='bilinear')(pooled)
        aspplist.append(pooled)      # (B, D, W, H)
        # fuse features
        out = torch.cat(aspplist, 1)
        out = self.conv2(self.relu(self.bn2(out)))
        return out

class DenseNet_atrous_aspp(nn.Module):
    """
    Dilation of the last densenet block is set as 2. Dilation rate of the ASPP is set as 6.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet_atrous_aspp, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=14, stride=3, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # First 3 denseblocks
        block_config_top = block_config[:3]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config_top):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config_top) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # 4th denseblock
        block = _DenseBlock_hdc(num_layers=block_config[-1], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, dilation=2)
        self.features.add_module('denseblock4', block)
        num_features = num_features + block_config[-1] * growth_rate

        # ASPP
        aspp = _ASPP(num_features, rate=6)
        self.features.add_module('aspp', aspp)
        num_features = 256

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.printsize = True

    def forward(self, x):
        # features = self.features(x)
        features = x
        for module_pos, module in self.features._modules.items():
            features = module(features)
            if self.printsize: print(module_pos, features.size())

        out = F.relu(features, inplace=True)
        if self.printsize: self.printsize = False
        out = F.avg_pool2d(out, kernel_size=out.size(3), stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out