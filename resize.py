import os
from PIL import Image, ImageChops
from tqdm import tqdm
from pose_map import new_posemap


# 사진 이미지를 최대한 공백을 줄이면서, 원래 모델에 입력하는 input의 비율에 맞게 3:4로 crop
def trim(mask_im):
    background = Image.new(mask_im.mode, mask_im.size, mask_im.getpixel((0, 0)))
    diff = ImageChops.difference(mask_im, background)
    diff = ImageChops.add(diff, diff, 12, -17)
    bbox = diff.getbbox()
    if bbox:
        ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        if ratio > 0.75:
            x1 = bbox[0] - 20
            x2 = bbox[2] + 20
            w = x2 - x1
            c = (bbox[3] - bbox[1]) / 2 + bbox[1]
            h = 4 / 3 * w
            y1 = c - (h / 2)
            y2 = c + (h / 2)
            bbox = (x1, y1, x2, y2)

        elif ratio < 0.75:
            y1 = bbox[1] - 20
            y2 = bbox[3] + 20
            h = y2 - y1
            c = (bbox[2] - bbox[0]) / 2 + bbox[0]
            w = 3 / 4 * h
            x1 = c - (w / 2)
            x2 = c + (w / 2)
            bbox = (x1, y1, x2, y2)

        return bbox

    else:
        print('Failure!')
        return None


# PIL Image 파일을 target_size로 resize 해주는 함수
def img_resize(im, target_size):
    return im.resize(target_size)


# image들을 주어진 bbox에 맞게 crop하고 목표 크기로 resizing
def crop_and_resize(image_list, bbox, target_size):
    crop_images = [image.crop(bbox) for image in image_list]
    return (img_resize(image, target_size) for image in crop_images)


# 리스트에 있는 이미지를 color(모델 이미지)와 마스크를 불러오고 크롭 이미지로 만들어 새 폴더에 저장하는 함수
def resize(
    image_path, mask_path,
    image_save, mask_save,

    seg_path=None, pose_path=None,
    seg_save=None, pose_save=None,
    # True for Item image process only
    is_item=False
):
    # resized image가 저장될 directory
    os.makedirs(image_save, exist_ok=True)
    os.makedirs(mask_save, exist_ok=True)

    # target_size 설정
    target_size = (192, 256)

    # Only resizing Item(Garment) Images and Mask
    if is_item:
        image_list, mask_list = sorted(os.listdir(image_path)), sorted(os.listdir(mask_path))

        for model_image, mask_image in tqdm(
            zip(image_list, mask_list),
            desc="Resizing all images and making new Pose map"
        ):
            im = Image.open(f'{image_path}/{model_image}')
            mask_im = Image.open(f'{mask_path}/{mask_image}')

            # get bbox
            bbox = trim(mask_im)

            # crop and resize
            resize_im, resize_mask_im = crop_and_resize([im, mask_im], bbox, target_size)

            # save resized images
            resize_im.save(f'{image_save}/{model_image}')
            resize_mask_im.save(f'{mask_save}/{mask_image}')

    # Model Image/Model Mask/Model segmentation resizing
    # Make new Pose map
    else:
        # resized seg_image, new pose map 저장할 디렉토리를 생성
        os.makedirs(seg_save, exist_ok=True)
        os.makedirs(pose_save, exist_ok=True)

        image_list, mask_list, seg_list, pose_list = \
            sorted(os.listdir(image_path)), sorted(os.listdir(mask_path)), \
            sorted(os.listdir(seg_path)), sorted(os.listdir(pose_path))

        for model_image, mask_image, seg_image, pose_json in tqdm(
            zip(image_list, mask_list, seg_list, pose_list),
            desc="Resizing all images and making new Pose map"
        ):
            im = Image.open(f'{image_path}/{model_image}')
            mask_im = Image.open(f'{mask_path}/{mask_image}')
            seg_im = Image.open(f'{seg_path}/{seg_image}')

            # get bbox
            bbox = trim(mask_im)

            # crop
            resize_im, resize_mask_im, resize_seg_im = \
                crop_and_resize([im, mask_im, seg_im], bbox, target_size)

            resize_ratio = resize_im.size[0] / im.size[0]
            # make new Pose map for resized images
            new_posemap(pose_path, pose_json, pose_save, bbox, resize_ratio)

            # save resized images
            resize_im.save(f'{image_save}/{model_image}')
            resize_mask_im.save(f'{mask_save}/{mask_image}')
            resize_seg_im.save(f'{seg_save}/{seg_image}')