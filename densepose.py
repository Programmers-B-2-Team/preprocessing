import torch
import torch.nn.functional as F
import pickle
import numpy as np
import cv2
import os


def make_densepose(pkl_path, image_path, save_path):
    with open(f"{pkl_path}./results.pkl", "rb") as f:
        data = pickle.load(f)
    os.makedirs(save_path, exist_ok=True)

    for i, item in enumerate(os.listdir(f'{image_path}')):
        res = data[i]["pred_densepose"][0].labels.to("cpu")

        # detectron2의 densepose 기능을 실행한 결과는 object detection을 포함하고 있다.
        # object detection을 통해 이미지 내의 물체 영역을 직사각형으로 제한하고, 이에 대한 densepose를 저장한다.
        # 따라서 실제 결과는 원본 이미지보다 이미지 사이즈가 작아지게 된다.
        # 이미지의 실제 크기를 원본 이미지의 배치와 가깝게 복원하기 위해서, 다음과 같은 방식으로 패딩을 추가해주도록 한다.
        image = cv2.imread(f'{image_path}/{item}')
        ori_height, ori_width = image.shape[0], image.shape[1]

        padding_v = torch.zeros((ori_height - res.shape[0]) // 2, res.shape[1])
        res = torch.cat([padding_v, res, padding_v])
        padding_h = torch.zeros(res.shape[0], (ori_width - res.shape[1]) // 2)
        res = torch.cat([padding_h, res, padding_h], 1)

        # 패딩 추가 후 목표 이미지 크기에 완전히 일치하도록 shape를 복원한다.
        res = res.reshape(1, 1, res.shape[0], res.shape[1]).to(torch.float32) # this tensor size is (1, 1, res.height, res.width)
        res = F.interpolate(res, (ori_height, ori_width))\
            .reshape(ori_height, ori_width).numpy().astype(np.uint8) # nearest neighbor interpolation

        # shape 복원은 interpolate와 upsample 방식이 있다
        # upsample = torch.nn.Upsample(size=(ori_height, ori_width))
        # res = upsample(res).reshape(ori_height, ori_width).numpy().astype(np.uint8)

        np.save(f"{save_path}/{item.split('.')[0]}.npy", res)
