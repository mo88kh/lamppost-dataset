from __future__ import annotations
import json, datetime, getpass
from pathlib import Path
import numpy as np
from .utils import TILE_SIZE

def save_labels(path: Path, labels: np.ndarray) -> None:
    rows, cols = labels.shape
    payload = {
        "tile_size": TILE_SIZE,
        "rows": int(rows),
        "cols": int(cols),
        "labels": labels.tolist(),
        "annotator": getpass.getuser(),
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)

def load_labels(path: Path) -> np.ndarray:
    meta = json.loads(path.read_text())
    return np.array(meta["labels"], dtype=np.int8)
