import json
import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage import img_as_ubyte
import skimage.io, skimage.color, skimage.filters
from PIL import Image


# grabcut based binary masking method
def cloth_masking_with_grabcut(im_path, mask_path):
    lo = 250
    hi = 255

    img = cv2.imread(im_path, 0)
    img2 = cv2.imread(im_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # 1. binary thresholding
    ret, th_bin = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY_INV)

    # 2. Filling operation:

    # 2.1 Copy the thresholded image.
    im_floodfill = th_bin
    # 2.2 Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th_bin.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # 2.3 Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    # 2.4 Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 2.5 Combine the two images to get the foreground.
    th_filled = th_bin | im_floodfill_inv

    # 3. Morphology operation:
    kernel = np.ones((2, 2), np.uint8)

    # 3.1 opening for salt noise removal
    th_opened = cv2.morphologyEx(th_filled, cv2.MORPH_OPEN, kernel)

    # 3.2 closing for pepper noise removal (not needed it seems)
    # th_closed = cv2.morphologyEx(th_opened, cv2.MORPH_CLOSE, kernel)

    # 3.3 erosion for thinning out boundary
    # kernel = np.ones((3, 3), np.uint8)
    # th_eroded = cv2.erode(th_opened, kernel, iterations=1)

    # 4. GrabCut

    # 4.1 make mask
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    gc_mask = np.zeros(img2.shape[:2], np.uint8)
    newmask = th_opened

    # 4.2 define GrabCut priors
    absolute_foreground = cv2.erode(newmask, kernel, iterations=2)
    probable_foreground = newmask - absolute_foreground
    dilated_newmask = cv2.dilate(newmask, kernel, iterations=2)
    absolute_background = cv2.bitwise_not(dilated_newmask)
    probable_background = dilated_newmask - newmask

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 4.3 change mask based on priors
    # any mask values greater than zero should be set to probable
    # foreground
    gc_mask[absolute_foreground > 0] = cv2.GC_FGD
    gc_mask[probable_foreground > 0] = cv2.GC_PR_FGD
    gc_mask[absolute_background > 0] = cv2.GC_BGD
    gc_mask[probable_background > 0] = cv2.GC_PR_BGD

    # 4.4 apply GrabCut masking/segmentation
    gc_mask, bgdModel, fgdModel = cv2.grabCut(img2, gc_mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    gc_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')

    # 6. save result
    gc_mask[gc_mask > 0] = 255    # make visible white
    cv2.imwrite(mask_path, gc_mask)


# image dir, json dir과 결과를 저장할 dir 경로, image-only 여부를 arguments로 제공
def binary_masking(image_path, json_path, save_path, image_only):
    # binary masked image file이 저장될 directory 생성
    os.makedirs(save_path, exist_ok=True)

    if image_only:
        image_list = os.listdir(image_path)
        for image in tqdm(image_list, desc="Binary Masking ..."):
            im_path = os.path.join(image_path, image)
            res_path = os.path.join(save_path, image)
            cloth_masking_with_grabcut(im_path, res_path)

    else:
        # 주어진 디렉토리 내의 이미지와 json file 순서를 동일하기 맞추기 위한 정렬과정
        image_list, json_list = sorted(os.listdir(image_path)), sorted(os.listdir(json_path))

        for image_file, json_file in tqdm(
            zip(image_list, json_list),
            desc="Binary Masking ..."
        ):
            img = cv2.imread(f'{image_path}/{image_file}')
            # 이미지 사이즈에 맞는 영행렬 생성
            base_image = np.zeros_like(img)
            json_data = json.load(open(f'{json_path}/{json_file}'))

            # segmentation 영역별로 빈 이미지 위에 반복해서 덮어 씌우기
            for item in json_data.keys():
                if 'region' in item:
                    pts = json_data[item]['segmentation']
                    for i in range(len(pts)):
                        area = np.array(pts[i], dtype=np.int32)
                        base_image = cv2.fillPoly(base_image, [area], (255, 255, 255))

            cv2.imwrite(f'{save_path}/{image_file}', base_image)

