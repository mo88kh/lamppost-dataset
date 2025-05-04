from pathlib import Path
import numpy as np
from src.annotator import io

def test_io_roundtrip(tmp_path: Path):
    lbl = np.array([[0, 1], [-1, 0]], dtype=np.int8)
    f = tmp_path / "lbl.json"
    io.save_labels(f, lbl)
    loaded = io.load_labels(f)
    assert np.array_equal(lbl, loaded)
