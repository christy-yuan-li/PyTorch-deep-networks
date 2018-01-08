"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np


def softmax(input, dim=1):
    # input (num_capsules, B, out_channels)
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

NUM_CLASSES = 2

class CapsuleLayer(nn.Module):
    def __init__(self, vector_len, num_capsules, num_route_nodes, in_channels, kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.vector_len = vector_len
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            # Use FC layer, route_weights is the transformation matrix
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, vector_len))
        else:
            # Use conv layer
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, vector_len, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])


    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        '''
        :param x: (B, in_channels, W, H) if from non-capsule-layer, (B, num_capsules_prev*W*H, vector_len_prev) if from capsule-layer
        :return: outputs: (B, num_capsules*W*H, vector_len) if not routing, (num_capsules, B, 1, 1, vector_len) if routing
        '''
        if self.num_route_nodes != -1:
            print(x.size(), x[None, :, :, None, :].size())
            print(self.route_weights.size(), self.route_weights[:, None, :, :, :].size())

            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]      # dot product, (10, B, 32*6*6, 1, 16)
            # convert in_channels vectors to out_channels vectors
            # x:    (B, num_capsules_prev*W*H, vector_len_prev)
            # route_weights:    (num_capsules, num_route_nodes, in_channels, vector_len)
            # (1, B, num_capsules_prev*W*H, 1, vector_len_prev) * (num_capsules, 1, num_route_nodes, in_channels, vector_len) =
            # (num_capsules, B, num_route_nodes, 1, vector_len)

            # initialize logits as zeros
            logits = Variable(torch.zeros(*priors.size())).cuda()   # coupling coefficients (num_capsules, B, num_route_nodes, 1, vector_len)

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)         # (num_capsules, B, num_route_nodes, 1, vector_len)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True)) # (num_capsules, B, 1, 1, vector_len)

                if i != self.num_iterations - 1:
                    # compute similarity
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)  # (num_capsules, B, num_route_nodes, 1, vector_len)
                    logits = logits + delta_logits

        else:
            outputs = [capsule(x).unsqueeze_(4) for capsule in self.capsules]  # [(B, vector_len, W, H)]
            print(outputs[0].size())
            outputs = torch.cat(outputs, dim=4)        # (B, vector_len, W, H, num_capsules)
            print(outputs.size())
            outputs = outputs.transpose(1, 4)    # (B, num_capsules, W, H, vector_len)
            print(outputs.size())
            outputs = outputs.contiguous().view(outputs.size(0), -1, outputs.size(-1))   # (B, num_capsules*W*H, vector_len)
            print(outputs.size())

        return outputs



class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=256,
                               kernel_size=9,
                               stride=1)    # x (B, C, W, H)

        self.primary_capsules = CapsuleLayer(vector_len=8,
                                           num_capsules=32,
                                           num_route_nodes=-1,
                                           in_channels=256,
                                           kernel_size=9,
                                           stride=2)       # (B, num_capsules*W*H, vector_len)

        self.digit_capsules = CapsuleLayer(vector_len=16,
                                         num_capsules=NUM_CLASSES,
                                         num_route_nodes=32 * 6 * 6,
                                         in_channels=8)  # (num_capsules, B, 1, 1, vector_len)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),   # 28 * 28 image
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)     # (B, C, W, H)
        x = self.primary_capsules(x)        # (B, num_capsules*W*H, out_channels)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)  # (num_capsules, B, 1, 1, vector_len) -> (B, num_capsules, vector_len)

        classes = (x ** 2).sum(dim=-1) ** 0.5   # (B, num_capsules), norm of the vector is the probability
        classes = F.softmax(classes)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions
