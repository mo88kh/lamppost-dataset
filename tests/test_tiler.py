import cv2
import numpy as np
from pathlib import Path
from src.tiler import tile_image   # <-- import from the module you wrote

def test_tile_image(tmp_path: Path):
    # synthetic black image 300 × 200 px
    dummy = np.zeros((200, 300, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(img_path), dummy)

    out_dir = tmp_path / "tiles"
    n = tile_image(img_path, out_dir, tile_size=100)

    assert n == 6                           # 2 rows × 3 cols
    assert len(list(out_dir.iterdir())) == 6
