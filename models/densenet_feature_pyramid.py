import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append('../')
import transforms
from models.layers import SplitSumTable, MaskLayer, RelationalUnit


__all__ = ['densenet121FPN']



def densenet121FPN(**kwargs):
    model = DenseNetFPN(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
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


class DenseNetFPN(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

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

        super(DenseNetFPN, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=14, stride=3, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        blockwise_num_features = []
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            blockwise_num_features.append(num_features) # for denseblock1-4

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Top layer
        self.toplayer_norm = nn.BatchNorm2d(blockwise_num_features[-1])
        self.toplayer = nn.Conv2d(blockwise_num_features[-1], 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1_norm = nn.BatchNorm2d(256)
        self.smooth2_norm = nn.BatchNorm2d(256)
        self.smooth3_norm = nn.BatchNorm2d(256)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_norm = nn.BatchNorm2d(blockwise_num_features[-2])
        self.latlayer2_norm = nn.BatchNorm2d(blockwise_num_features[-3])
        self.latlayer3_norm = nn.BatchNorm2d(blockwise_num_features[-4])
        self.latlayer1 = nn.Conv2d(blockwise_num_features[-2], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(blockwise_num_features[-3], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(blockwise_num_features[-4], 256, kernel_size=1, stride=1, padding=0)

        # Final batch norm for each scale features
        self.p5_norm = nn.BatchNorm2d(256)
        self.p4_norm = nn.BatchNorm2d(256)
        self.p3_norm = nn.BatchNorm2d(256)
        self.p2_norm = nn.BatchNorm2d(256)

        # Linear layer
        self.classifier = nn.Linear(256 * 4, num_classes)
        self.printsize = True

    def forward(self, x):
        # implementation of feature pyramid network
        layerwise_features = []
        features = x
        for module_pos, module in self.features._modules.items():
            insize = features.size()
            features = module(features)
            if self.printsize: print(module_pos, insize, features.size())
            if 'denseblock' in module_pos:
                layerwise_features.append(features)
        assert(len(layerwise_features) == 4)

        p5 = self.toplayer(F.relu(self.toplayer_norm(layerwise_features[-1]), inplace=True))     # 1*1, ->256
        p4 = self._upsample_add(p5, self.latlayer1(F.relu(self.latlayer1_norm(layerwise_features[-2]), inplace=True)))     # 1*1, ->256
        p3 = self._upsample_add(p4, self.latlayer2(F.relu(self.latlayer2_norm(layerwise_features[-3]), inplace=True)))    # 1*1, ->256
        p2 = self._upsample_add(p3, self.latlayer3(F.relu(self.latlayer3_norm(layerwise_features[-4]), inplace=True)))    # 1*1, ->256
        # Smooth
        p4 = self.smooth1(F.relu(self.smooth1_norm(p4), inplace=True))   # 3*3, ->256
        p3 = self.smooth2(F.relu(self.smooth2_norm(p3), inplace=True))   # 3*3, ->256
        p2 = self.smooth3(F.relu(self.smooth3_norm(p2), inplace=True))   # 3*3, ->256

        p5 = F.relu(self.p5_norm(p5), inplace=True)
        p4 = F.relu(self.p4_norm(p4), inplace=True)
        p3 = F.relu(self.p3_norm(p3), inplace=True)
        p2 = F.relu(self.p2_norm(p2), inplace=True)

        if self.printsize:
            print('p2', p2.size())
            print('p3', p3.size())
            print('p4', p4.size())
            print('p5', p5.size())

        p5_avgpool = F.avg_pool2d(p5, kernel_size=p5.size(3), stride=1).view(p5.size(0), -1)
        p4_avgpool = F.avg_pool2d(p4, kernel_size=p4.size(3), stride=1).view(p4.size(0), -1)
        p3_avgpool = F.avg_pool2d(p3, kernel_size=p3.size(3), stride=1).view(p3.size(0), -1)
        p2_avgpool = F.avg_pool2d(p2, kernel_size=p2.size(3), stride=1).view(p2.size(0), -1)

        if self.printsize:
            print('p2_avgpool', p2_avgpool.size())
            print('p3_avgpool', p3_avgpool.size())
            print('p4_avgpool', p4_avgpool.size())
            print('p5_avgpool', p5_avgpool.size())

        out = torch.cat([p2_avgpool, p3_avgpool, p4_avgpool, p5_avgpool], 1)
        if self.printsize:
            print('final features output size: ', out.size())
            self.printsize = False

        out = self.classifier(out)
        return out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

