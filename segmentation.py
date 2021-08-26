import json
import cv2
import numpy as np
import os
from tqdm import tqdm


def colormap(n=21):
    """
    Default Value : CP VTON+ segmentation label
    해당 설정은 PF-AFN 논문에서 사용한 label과 다르므로 논문의 학습 코드를
    일부 수정해야 한다.
    0=Background
    1=Hat
    2=Hair
    3=Glove
    4=SunGlasses
    5=UpperClothes
    6=Dress
    7=Coat
    8=Socks
    9=Pants
    10=Jumpsuits
    11=Scarf
    12=Skirt
    13=Face
    14=LeftArm
    15=RightArm
    16=LeftLeg
    17=RightLeg
    18=LeftShoe
    19=RightShoe
    20=Skin/Neck/Chest
    """
    cmap = [(i, i, i) for i in range(n)]
    return cmap


# 모델 image dir, json dir과 segmentaion 결과를 저장할 dir 경로를 arguments로 제공
def segmentation(image_path, json_path, save_path):
    os.makedirs(save_path, exist_ok=True)

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

    cmap = colormap()
    # 주어진 디렉토리 내의 이미지와 json file 순서를 동일하기 맞추기 위한 정렬과정
    image_list, json_list = sorted(os.listdir(image_path)), sorted(os.listdir(json_path))

    for image_file, json_file in tqdm(
        zip(image_list, json_list),
        desc="Segmentation Labeling ..."
    ):
        model_image = cv2.imread(f'{image_path}/{image_file}')
        # 이미지 사이즈에 맞는 영행렬 생성
        base_image = np.zeros_like(model_image)
        json_data = json.load(open(f'{json_path}/{json_file}'))

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

            cv2.imwrite(f'{save_path}/{image_file}', base_image)

        # else:
        #     cv2.imwrite(f'{save_path}/{image_file}', base_image)
