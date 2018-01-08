import torch
import math
irange = range
from PIL import Image
import numpy as np
import os
import cv2
import shutil
from sklearn.metrics import roc_curve, auc
import functools

def resume_checkpoint(model, checkpoint_path, metric):
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        # Load part of the model
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and ('classifier' not in k)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        return checkpoint['epoch'], checkpoint[metric]
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        return 0, 0


def resume_pretrained(model, checkpoint_path, metric):
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        # Load part of the model
        pretrained_dict = checkpoint
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and ('classifier' not in k) and ('features.norm5' not in k)}
        # 2. overwrite entries in the existing state dict
        for key in pretrained_dict.keys():
            print(key, pretrained_dict[key].size())

        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, 0))
        return 0, 0
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        return 0, 0


def save_checkpoint(state, is_best, exp_name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (exp_name) + 'model_best.pth.tar')

def macro_auc(output, target):
    classes = len(target[0])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    macroauc = 0.0
    count = 0
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        if not math.isnan(roc_auc[i]):  # if there is no positive sample of this class, skip evaluating this class
            macroauc += roc_auc[i]
            count += 1
    if count > 0:
        macroauc /= float(count)
    return macroauc


def get_f1(confusion_matrix, beta=None):
    '''
    :param confusion_matrix:
    :param beta:
    :return: F1 value, 1-d array if num_classes > 2; a scalar value if num_classes == 2 for binary classification
    '''
    confusion_matrix = np.array(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = functools.reduce(lambda x, y: x + sum(y), confusion_matrix, 0) - (FP + FN + TP)

    avoid_nan = np.array([0.00001] * len(FP))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN + avoid_nan) * 100.00
    # Specificity or true negative rate
    TNR = TN / (TN + FP + avoid_nan) * 100.00
    # Precision or positive predictive value
    PPV = TP / (TP + FP + avoid_nan) * 100.00
    # Negative predictive value
    NPV = TN / (TN + FN + avoid_nan) * 100.00
    # Fall out or false positive rate
    FPR = FP / (FP + TN + avoid_nan) * 100.00
    # False negative rate
    FNR = FN / (TP + FN + avoid_nan) * 100.00
    # False discovery rate
    FDR = FP / (TP + FP + avoid_nan) * 100.00

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN + avoid_nan) * 100.00
    F1 = 2 * PPV * TPR / (PPV + TPR + avoid_nan)

    if len(confusion_matrix) == 2:
        return F1[1]
    else:
        return F1

def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid

