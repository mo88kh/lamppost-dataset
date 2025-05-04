from __future__ import annotations
import numpy as np
from typing import Tuple

TILE_SIZE = 100           # keep in sync with config.yaml

def image_to_grid(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Return (rows, cols) for a given image H×W shape."""
    h, w = shape
    return h // TILE_SIZE, w // TILE_SIZE

def pixel_to_cell(x: int, y: int) -> Tuple[int, int]:
    """Convert pixel coords to integer (row, col) of grid cell."""
    return y // TILE_SIZE, x // TILE_SIZE

def blank_label_matrix(rows: int, cols: int) -> np.ndarray:
    """Create an int8 matrix initialised to 0 (unlabelled)."""
    return np.zeros((rows, cols), dtype=np.int8)
