# CLI를 통해 arguments를 받아 Preprocess를 진행
# Typer를 통한 CLI 구현 시도
# 1) Binary Masking image(with JSON file)
# 2) Binary Masking image(without JSON file)
# 3) Segmentation Labeling image
# 4) Pose map JSON file(Training Code에서 일부 변경할 것)
# 5) Densepose Map npy file
import typer
from pathlib import Path
from typing import List
from masking import *


app = typer.Typer()


@app.command()
def binary_json(
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
def test():
    pass


if __name__ == "__main__":
    app()