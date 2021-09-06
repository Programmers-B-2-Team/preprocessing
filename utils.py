import os
import shutil
from os import walk
import numpy as np
import cv2
import torch
import re
from typing import Any, Dict, Tuple, Union
from pathlib import Path


def get_file_list(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
    return files


def move_listed_files(old_path,new_path, lst):
    os.makedirs(new_path, exist_ok=True)
    for i in lst:
        shutil.copy(old_path+ "/" + i, new_path)


# 모델 파일명의 마지막 숫자는 카메라를 기준으로 모델 포즈의 각도를 의미하는 것 (데이터 명세서에 따르면)
# 정면 사진만 활용할 것이기 때문에 "000.jpg"로 끝나는 파일이 아니면, 모두 삭제
# 특정 스트링이 포함된 리스트아이템을 모아 리스트를 만든다.
def new_list_certain_str(lst, string):
    new_lst = [item for item in lst if string in item]
    return new_lst


# binary_masking with U Net utilities
def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def load_rgb(image_path: Union[Path, str]) -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
    Returns: 3 channel array with RGB image
    """
    if Path(image_path).is_file():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_REFLECT_101) -> tuple:
    """Pads the image on the sides, so that it will be divisible by factor.
    Common use case: UNet type architectures.
    Args:
        image:
        factor:
        border: cv2 type border.
    Returns: padded_image
    """
    height, width = image.shape[:2]

    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(image: np.array, pads: Tuple[int, int, int, int]) -> np.ndarray:
    """Crops patch from the center so that sides are equal to pads.
    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    Returns: cropped image
    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
    height, width = image.shape[:2]

    return image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]