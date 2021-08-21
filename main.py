# 3) Segmentation Labeling image
# 4) Pose map JSON file(Training Code에서 일부 변경할 것)
# 5) Densepose Map npy file

import typer
from pathlib import Path
from typing import List
from masking import binary_masking
from segmentation import segmentation

app = typer.Typer()


@app.command()
def binary_mask(
    image_path: Path = typer.Argument(..., help="Garment Image file directory"),
    json_path: Path = typer.Option(None, help="Garment JSON files directory"),
    save_path: Path = typer.Option(Path('./masked'), help="Directory where result images saved in"),
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
    save_path: Path = typer.Option(Path('./segmentation'), help="Directory where result images saved in"),
):
    """
        Create Binary masked image(with JSON file)
        """
    segmentation(image_path, json_path, save_path)


if __name__ == "__main__":
    app()
