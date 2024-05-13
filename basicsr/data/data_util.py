import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from basicsr.data.transforms import mod_crop
from basicsr.utils import scandir


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths

def paired_DP_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, (
        'The len of folders should be 3 with [inputL_folder, inputR_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 2 with [inputL_key, inputR_key, gt_key]. '
        f'But got {len(keys)}')
    inputL_folder, inputR_folder, gt_folder = folders
    inputL_key, inputR_key, gt_key = keys

    inputL_paths = list(scandir(inputL_folder))
    inputR_paths = list(scandir(inputR_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(inputL_paths) == len(inputR_paths) == len(gt_paths), (
        f'{inputL_key} and {inputR_key} and {gt_key} datasets have different number of images: '
        f'{len(inputL_paths)}, {len(inputR_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        inputL_path = inputL_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputL_path))
        inputL_name = f'{filename_tmpl.format(basename)}{ext_input}'
        inputL_path = osp.join(inputL_folder, inputL_name)
        assert inputL_name in inputL_paths, (f'{inputL_name} is not in '
                                           f'{inputL_key}_paths.')
        inputR_path = inputR_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputR_path))
        inputR_name = f'{filename_tmpl.format(basename)}{ext_input}'
        inputR_path = osp.join(inputR_folder, inputR_name)
        assert inputR_name in inputR_paths, (f'{inputR_name} is not in '
                                           f'{inputR_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{inputL_key}_path', inputL_path),
                  (f'{inputR_key}_path', inputR_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
        0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
