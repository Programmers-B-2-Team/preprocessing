import json
import cv2
import numpy as np
import os
from tqdm import tqdm


label_colours = [
    (0, 0, 0),  # 0=Background
    (128, 0, 0),  # 1=Hat
    (255, 0, 0),  # 2=Hair
    (0, 85, 0),   # 3=Glove
    (170, 0, 51),  # 4=Sunglasses
    (255, 85, 0),  # 5=UpperClothes
    (0, 0, 85),  # 6=Dress
    (0, 119, 221),  # 7=Coat
    (85, 85, 0),  # 8=Socks
    (0, 85, 85),  # 9=Pants
    (85, 51, 0),  # 10=Jumpsuits
    (52, 86, 128),  # 11=Scarf
    (0, 128, 0),  # 12=Skirt
    (0, 0, 255),  # 13=Face
    (51, 170, 221),  # 14=LeftArm
    (0, 255, 255),  # 15=RightArm
    (85, 255, 170),  # 16=LeftLeg
    (170, 255, 85),  # 17=RightLeg
    (255, 255, 0),  # 18=LeftShoe
    (255, 170, 0),  # 19=RightShoe
    (189, 183, 107)  # 20=Neck    # new added
    ]


# AI Hub data segmentation label
color_dict = {
    "hat": 1,
    "hair": 2,
    "face": 13,
    "neck": 20,

    "inner_torso": 5,
    "inner_rsleeve": 5,
    "inner_lsleeve": 5,
    "dress_torso": 5,
    "dress_rsleeve": 5,
    "dress_lsleeve": 5,

    "outer_torso": 5,  # 데이터 일관성 위해 임시 추가
    "outer_lsleeve": 5,  # 데이터 일관성 위해 임시 추가
    "outer_rsleeve": 5,  # 데이터 일관성 위해 임시 추가

    "pants_hip": 9,
    "pants_rsleeve": 9,
    "pants_lsleeve": 9,
    "skirt": 9,

    "left_arm": 14,
    "right_arm": 15,

    "left_leg": 16,
    "right_leg": 17,
    "left_shoe": 18,
    "right_shoe": 19
    }


# 모델 image dir, json dir과 segmentaion 결과를 저장할 dir 경로를 arguments로 제공
def segmentation(json_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    # cmap = colormap()
    cmap = label_colours
    # 주어진 디렉토리 내의 이미지와 json file 순서를 동일하기 맞추기 위한 정렬과정
    json_list = sorted(os.listdir(json_path))

    for json_file in tqdm(
        json_list,
        desc="Segmentation Labeling ..."
    ):
        json_data = json.load(open(f'{json_path}/{json_file}'))
        # 이미지 사이즈에 맞는 영행렬 생성
        base_image = np.zeros([json_data['image_size']['height'], json_data['image_size']['width'], 3], dtype=np.uint8)
        name = json_file.split('.')[0]

        # region에 따라 매핑된 컬러로 좌표 영역에 컬러 이미지 생성 및 덮어쓰기 반복
        for item in json_data.keys():
            if "region" in item:
                data = json_data[item]
                category = data['category_name']
                # outer는 제외하고 학습시킨다
                # 만약 사전에 데이터셋에서 outer를 제외했다면 for-else 문을 삭제한다
                # if "outer" in category:
                    # break

                color = cmap[color_dict[category]]
                pts = data['segmentation']
                for i in range(len(pts)):
                    area = np.array(pts[i], dtype=np.int32)
                    base_image = cv2.fillPoly(base_image, [area], color)

        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{save_path}/{name}.png', base_image)

        # else:
        #     cv2.imwrite(f'{save_path}/{image_file}', base_image)


def indexed_segmentation(json_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    json_list = sorted(os.listdir(json_path))

    for json_file in tqdm(
        json_list,
        desc="Segmentation Labeling ..."
    ):
        json_data = json.load(open(f'{json_path}/{json_file}'))
        # 이미지 사이즈에 맞는 2차원 영행렬 생성
        base_image = np.zeros([json_data['image_size']['height'], json_data['image_size']['width']], dtype=np.uint8)
        name = json_file.split('.')[0]

        for item in json_data.keys():
            if "region" in item:
                data = json_data[item]
                category = data['category_name']
                # label 그 자체가 color로서 사용된다
                color = color_dict[category]
                pts = data['segmentation']
                for i in range(len(pts)):
                    area = np.array(pts[i], dtype=np.int32)
                    base_image = cv2.fillPoly(base_image, [area], color)

        cv2.imwrite(f'{save_path}/{name}.png', base_image)
