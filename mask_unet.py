from collections import namedtuple
from torch import nn
from torch.utils import model_zoo
from segmentation_models_pytorch import Unet
import albumentations as albu
from utils import *

model = namedtuple("model", ["url", "model"])
models = {
    "Unet_2020-10-30": model(
        url="https://github.com/ternaus/cloths_segmentation/releases/download/0.0.1/weights.zip",
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model


# U Net을 사용한 binary masking. 정확도는 최상, 속도가 느리다는 단점
def unet_masking(image_path, save_path):
    # binary masked image file이 저장될 directory 생성
    os.makedirs(save_path, exist_ok=True)

    model = create_model("Unet_2020-10-30")
    transform = albu.Compose([albu.Normalize(p=1)], p=1)

    image_list = os.listdir(image_path)

    for image in image_list:
        im = load_rgb(f'{image_path}/{image}')
        padded_image, pads = pad(im, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

        with torch.no_grad():
            prediction = model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        masked_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255

        cv2.imwrite(f'{save_path}/{image}', masked_image)