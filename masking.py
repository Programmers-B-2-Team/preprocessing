import json
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from skimage import img_as_ubyte
import skimage.io, skimage.color, skimage.filters


# 옷 image dir, json dir과 결과를 저장할 dir 경로를 arguments로 제공
def binary_masking(image_path, json_path, save_path, image_only):
    # binary masked image file이 저장될 directory 생성
    os.makedirs(save_path, exist_ok=True)

    if image_only:
        image_list = os.listdir(image_path)
        for filename in tqdm(image_list):
            sigma = 2.0
            t = 0.8
            image = skimage.io.imread(fname=f'{image_path}/{filename}')
            gray = skimage.color.rgb2gray(image)
            blur = skimage.filters.gaussian(gray, sigma=sigma)

            # perform inverse binary thresholding
            mask = blur < t
            # save
            skimage.io.imsave(
                f'{save_path}/{filename}',
                img_as_ubyte(mask),
                check_contrast=False
            )
    else:
        # 주어진 디렉토리 내의 이미지와 json file 순서를 동일하기 맞추기 위한 정렬과정
        image_list, json_list = sorted(os.listdir(image_path)), sorted(os.listdir(json_path))

        for image_file, json_file in tqdm(
            zip(image_list, json_list),
            desc="Binary Masking ..."
        ):
            img = cv.imread(f'{image_path}/{image_file}')
            # 이미지 사이즈에 맞는 영행렬 생성
            base_image = np.zeros_like(img)
            json_data = json.load(open(f'{json_path}/{json_file}'))

            # segmentation 영역별로 빈 이미지 위에 반복해서 덮어 씌우기
            for item in json_data.keys():
                if 'region' in item:
                    pts = json_data[item]['segmentation']
                    for i in range(len(pts)):
                        area = np.array(pts[i], dtype=np.int32)
                        base_image = cv.fillPoly(base_image, [area], (255, 255, 255))

            cv.imwrite(f'{save_path}/{image_file}', base_image)

