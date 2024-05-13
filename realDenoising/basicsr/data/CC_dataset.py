import os
import os.path as osp
import sys
# sys.path.append("/data/czx/AECC")
from typing import Tuple
from torchvision.transforms import Resize, ToTensor
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import torch
import torch.utils.data as data
from basicsr.utils.img_util import bgr_to_rgb, linear_to_nonlinear, hwc_to_chw
from basicsr.utils.DataAugmenter import DataAugmenter


class ColorCheckerDataset(data.Dataset):

    def __init__(self, opt, train):

        self.__train = train
        self.__opt = opt
        self.__da = DataAugmenter()
        self.__path_to_metadata = opt["dataroot_label"]
        self.__path_to_data = opt["dataroot_img"]
        self.__path_to_gt = opt["dataroot_gt"]
        metadata = open(self.__path_to_metadata, 'r').read().splitlines()
        metadata_name  = []
        metadata_label = []
        for row in metadata:
            metadata_name.append(row.split(" ")[1])
            metadata_label.append([float(row.split(" ")[2]),float(row.split(" ")[3]),float(row.split(" ")[4])])
        img_name = os.listdir(self.__path_to_data)
        img_idx = []
        for row in img_name:
            img_idx.append(metadata_name.index(row))
        
        self.__fold_data = [metadata[i] for i in img_idx]

    def __getitem__(self, index: int) -> Tuple:
        file_name = self.__fold_data[index].strip().split(' ')[1]
        img = cv2.imread(osp.join(self.__path_to_data, file_name))
        label = [float(self.__fold_data[index].strip().split(' ')[2]),float(self.__fold_data[index].strip().split(' ')[3]),float(self.__fold_data[index].strip().split(' ')[4])]
        gt = cv2.imread(osp.join(self.__path_to_gt,file_name))
        # img = np.array(np.load(os.path.join(self.__path_to_data, file_name + '.npy')), dtype='float32')
        # label = np.array(np.load(os.path.join(self.__path_to_label, file_name + '.npy')), dtype='float32')
        # gt = np.array(np.load(os.path.join(self.__path_to_gt, file_name + '.npy')), dtype='float32')
        
        img = hwc_to_chw(bgr_to_rgb(img))
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(np.array(label))
        gt = torch.from_numpy(gt)
        
        torch_resize = Resize(self.__opt.img_size)
        img = torch_resize(img)
        gt = torch_resize(gt)
        if not self.__train:
            img = img.type(torch.FloatTensor)

        return img, label, gt, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
