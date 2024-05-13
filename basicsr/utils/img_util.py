import cv2
import math
import numpy as np
import os
import torch
from PIL.Image import Image
from torch import Tensor
import torchvision.transforms.functional as F
from typing import Union, List, Tuple
from basicsr.utils.settings import DEVICE
from torch.nn.functional import interpolate



def correct(img_without_mcc:np.ndarray, img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).cuda(device = DEVICE[0])
    img_without_mcc = bgr_to_rgb(normalize(img_without_mcc))
    img_without_mcc=torch.from_numpy(np.array(hwc_to_chw(img_without_mcc))).cuda(device = DEVICE[0])
    # print(img.shape) # [3, H, W]
    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).cuda(device = DEVICE[0])
    # torch_resize = Resize([1024,1024])
    # correction = torch_resize(correction)*torch.eye(1024).cuda(device = DEVICE[0])
    # img_without_mcc = torch_resize(img_without_mcc)
    # correction_inverse = torch.linalg.inv(correction)
    
    corrected_img = torch.div(img_without_mcc, correction + 1e-10)
    # corrected_img = correction_inverse.float()@img_without_mcc
    F.to_pil_image(linear_to_nonlinear(corrected_img.squeeze()), mode="RGB").save("corr.jpg")
    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # print(max_img.shape) # [1,1,1,1]
    normalized_img = torch.div(corrected_img, max_img).squeeze()
    # print(normalized_img.shape) # [1,C,H,W]
    return F.to_pil_image(linear_to_nonlinear(normalized_img), mode="RGB"), corrected_img.cpu().numpy().squeeze()


# gamma校正
def linear_to_nonlinear(img: Union[np.array, Image, Tensor]) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / 2.2))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / 2.2)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / 2.5).squeeze(), mode="RGB")


# def normalize(img: np.ndarray) -> np.ndarray:
#     max_int = 65535.0
#     return np.clip(img, 0.0, max_int) * (1.0 / max_int)


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def bgr_to_rgb(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1]


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    """ Converts an image from height x width x channels to channels x height x width """
    return x.transpose(2, 0, 1)

def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    return x.transpose(1, 2, 0)


def scale(x: Tensor) -> Tensor:
    """ Scales all values of a tensor between 0 and 1 """
    x = x - x.min()
    x = x / x.max()
    return x


def rescale(x: Tensor, size: Tuple) -> Tensor:
    """ Rescale tensor to image size for better visualization """
    return interpolate(x, size, mode='bilinear')


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle).item()