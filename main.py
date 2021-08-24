import os
import typer
from pathlib import Path
from densepose import make_densepose
from masking import binary_masking
from resize import resize
from segmentation import segmentation


app = typer.Typer()


@app.command()
def binary_mask(
    image_path: Path = typer.Argument(..., help="Garment/Model Image file directory"),
    save_path: Path = typer.Argument(..., help="Directory where result images saved in"),
    json_path: Path = typer.Option(None, help="Garment/Model JSON files directory"),
    image_only: bool = typer.Option(False, help="Check Garment/Model JSON files exist")
):
    """
    Create Binary masked image(with JSON file)
    """
    binary_masking(image_path, json_path, save_path, image_only)


@app.command()
def segmentation_label(
    image_path: Path = typer.Argument(..., help="Model Image file directory"),
    json_path: Path = typer.Argument(..., help="Model JSON files directory"),
    save_path: Path = typer.Option(Path('./model_segmentation'), help="Directory where result images saved in"),
):
    """
        Create Binary masked image(with JSON file)
        """
    segmentation(image_path, json_path, save_path)


@app.command()
def resize_and_pose(
    image_path: Path = typer.Argument(..., help="Model Image file directory"),
    mask_path: Path = typer.Argument(..., help="Model Mask files directory"),
    model_image_save: Path = typer.Option(Path('./resize_image'), help="Directory where resized images saved in"),
    model_mask_save: Path = typer.Option(Path('./resize _mask'), help="Directory where resized images saved in"),

    seg_path: Path = typer.Option(None, help="Segmenation files directory"),
    pose_path: Path = typer.Option(None, help="Pose JSON file directory"),
    seg_save: Path = typer.Option(Path('./resize_segmentation'), help="Directory where resized images saved in"),
    pose_save: Path = typer.Option(Path('./resize_pose'), help="Directory where new Pose JSON saved in"),

    is_item: bool = typer.Option(False, help="Check True for Item image resize only")
):
    """
    Resizing All Image files and Making new Pose map JSON data
    만약 의류 아이템 이미지만 리사이즈할 경우, 이미지 및 마스크 path만 argument로 제공하고, is_item을 True로 설정
    Model Image, Model Mask, Segmentation, Pose data를 처리할 경우 is_item을 False로 설정한 후,
    seg_path와 pose_path를 반드시 인자로 제공해야만 한다.
    """
    resize(
        image_path, mask_path,
        model_image_save, model_mask_save,
        seg_path, pose_path,
        seg_save, pose_save,
        is_item
    )


@app.command()
def densepose_map(
    detectron_dir: Path = typer.Argument('./detectron2/projects/DensePose', help="Detectron2 framework directory"),
    image_path: Path = typer.Argument(..., help="Model Image file directory"),
    save_path: Path = typer.Option(Path('./model_densepose'), help="Directory where result images saved in"),
):
    current_path = os.getcwd()
    image_abspath = os.path.abspath(image_path)

    os.chdir(f'{detectron_dir}')
    os.system(
        f'''python apply_net.py dump ./configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml
        model_final_844d15.pkl {image_abspath}/"*.jpg" --output results.pkl -v'''
    )
    pkl_path = os.getcwd()

    os.chdir(current_path)
    make_densepose(pkl_path, image_path, save_path)


if __name__ == "__main__":
    app()
