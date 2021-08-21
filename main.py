import os
import typer
from pathlib import Path
from densepose import make_densepose
from masking import binary_masking
from segmentation import segmentation


app = typer.Typer()


@app.command()
def binary_mask(
    image_path: Path = typer.Argument(..., help="Garment Image file directory"),
    json_path: Path = typer.Option(None, help="Garment JSON files directory"),
    save_path: Path = typer.Option(Path('./item_mask'), help="Directory where result images saved in"),
    image_only: bool = typer.Option(False, help="Check garment JSON files exist")
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
