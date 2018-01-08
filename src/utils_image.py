import torch
import math
irange = range
from PIL import Image
import numpy as np
import os
import cv2
import shutil
from sklearn.metrics import roc_curve, auc


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def overlay_masks(img, masks, dir=None, imgpath=None):
    '''
    :param img: PIL image or numpy array
    :param masks: a list of PIL image or 2d float numpy array not normalized
    :param dir:
    :param imgpath:
    :return: a list of img_with_heatmap (3d float numpy array scaled to 0-1)
    '''
    imgs = []
    if type(masks[0]) == np.ndarray:
        masks = [mask.squeeze() for mask in masks]  # make sure masks contain a list of 2d numpy arrays

    for i, mask in enumerate(masks):
        imgname = '{}_{}'.format(imgpath.split('/')[-1].split('.')[0], str(i))
        imgs.append(overlay_mask(img, mask, dir=dir, imgname=imgname))
    return imgs

def overlay_mask(img, mask, dir=None, imgname=None):
    '''
    :param img: PIL image or numpy array
    :param mask: PIL image or 2d float numpy array
    :param dir:
    :param imgname:
    :return: 3d float numpy array scaled to 0-1
    '''
    # process mask
    img = np.float32(img)
    mask = np.float32(mask)
    if img.shape[1:] != mask.shape:
        mask = cv2.resize(mask, img.shape[1:])
    mask = np.maximum(mask, 0)
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  # Normalize between 0-1
    mask = np.uint8(mask * 255)     # Scale between 0-255 to visualize
    # print("mask ", mask.shape, np.max(mask), np.min(mask))
    # print(' '.join(str(a) for a in mask[np.nonzero(mask)]))

    # Grayscale mask
    path_to_file = os.path.join(dir, imgname + '_Grayscale.jpg')
    cv2.imwrite(path_to_file, mask)

    # Heatmap of activation map
    mask_heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HSV)
    path_to_file = os.path.join(dir, imgname + '_Heatmap.jpg')
    cv2.imwrite(path_to_file, mask_heatmap)
    # print("mask_heatmap ", mask_heatmap.shape, np.max(mask_heatmap), np.min(mask_heatmap))

    # process img
    img = np.uint8(np.float32(img) * 255)
    img = img.transpose((1,2,0))
    # path_to_file = os.path.join(os.getcwd(), dir, imgname + '_img.jpg')
    # cv2.imwrite(path_to_file, img)

    # Heatmap on picture
    img_with_heatmap = np.float32(mask_heatmap) + np.float32(img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)  # Normalize between 0-1
    img_with_heatmap = np.uint8(255 * img_with_heatmap)     # Scale between 0-255 to visualize
    # print("img_with_heatmap ", img_with_heatmap.shape, np.max(img_with_heatmap), np.min(img_with_heatmap))

    path_to_file = os.path.join(os.getcwd(), dir, imgname +'_Heatmap_On_Image.jpg')
    # print('save img_with_heatmap to {}'.format(path_to_file))
    cv2.imwrite(path_to_file, img_with_heatmap)
    return img_with_heatmap

def convert_to_grayscale(cv2im):
    """
    Converts 3d image to grayscale
    :param cv2im: (numpy arr) RGB image with shape (D,W,H)
    :return grayscale_im: (numpy_arr): Grayscale image with shape (W,H)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)) # Normalize between 0-1
    return grayscale_im

def save_masks(mask, output_path, file_name, verbose=False):
    mask = np.float32(mask)
    mask = np.maximum(mask, 0)
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  # Normalize between 0-1
    mask = np.uint8(mask * 255)
    save_png(mask, os.path.join(output_path, file_name), mode='grayscale', verbose=verbose)


def save_png(img, save_path, mode='rgb', verbose=False):
    """
    img: [height, width, num_channels] np.uint8 array, where num_channels = 1
    (grayscale) or 3 (rgb).
    """
    if mode == 'grayscale':
        Image.fromarray(np.squeeze(img)).save(save_path)
    else:
        Image.fromarray(img, 'RGB').save(save_path)
    if verbose:
        print('Save image to', save_path)

